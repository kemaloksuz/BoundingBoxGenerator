import torch
import math
import numpy as np
from matplotlib import path
import pdb

class BoxSampler(object):
    def __init__(self, 
                 RoI_number=1, 
                 IoU_bin_bases=torch.tensor([0.73,0.12,0.15,0.05,0], dtype=torch.float),
                 IoU_weights=torch.tensor([0.5,0.6,0.7,0.8,0.9], dtype=torch.float),
                 IoU_limit_precision=1e-5):
        super(BoxSampler,self).__init__() 
        '''
        INPUTS:
        RoI_number : Number of RoIs/boxes to generate
        IoU_bin_bases  : N dimensional tensor storing the lower bounds for the bins.
                     Ex.[0.5, 0.6, 0.7, 0.8, 0.9] then there are 5 bins from [0.5,0.6] to [0.9, 1.0]
        IoU_weights: N dimensional tensor storing the weights of the bins. 
        IoU_limit_precision: While drawing the limits for an IoU (e.g. see Fig.2 red curves),  
                             it show the precision of the points. This is the part that makes the 
                             algorithm a bit slower and needs an improvement.            
        '''
        self.RoI_number=RoI_number
        self.IoU_bin_bases=IoU_bin_bases
        self.IoU_weights=IoU_weights
        self.IoU_limit_precision=IoU_limit_precision
        self.IoU_bin_tops=torch.cat([IoU_bin_bases[1:], torch.tensor([1.])])
        self.bin_width=self.IoU_bin_tops-self.IoU_bin_bases

        # We assume that self.reference_box is a square. Following coordinates are preferred
        # since even the IoU=0.5, the limits will be always positive (see Fig.2 or Fig.6 in the paper).
        self.reference_box=[0.3, 0.3, 0.6, 0.6]

    def isnan(self,x):
        return x != x

    def sample_single(self, B, IoUs, imgSize):
        '''
        Samples a set of bounding boxes for a given input BB.

        INPUTS:
        B        : Input BB (i.e. B in Alg.1 in the paper) Mx4 dimensional tensor. 
                          A BB is represented by [TL_x, TL_y, BR_x, BR_y]
        IoUs            : Set of IoU thresholds. T in Alg.1. A box is generated for each IoU.
        imgSize         : [width, height] of the image. Ensures that the generated box is in the image.   
        '''

        #Normalize the input box such that it is shifted/scaled on the reference box
        #that resides at [0.3, 0.3, 0.6, 0.6]. Save scale and shift, for renormalization 
        #before returning. All operations are conducted within [0, 1] range. Hence we do not,
        #normalize image, we normalize the boxes owing to Theorems 1 and 2 in the paper.
        inputBox, scale, shift=self.normalize(B.clone().detach().unsqueeze(0)) 

        #BoundingBoxGenerator is doing exactly what Alg.1 in the paper achieves.
        #Given a GT/input BB and and IoU, it generates the boxes with the desired IoU.
        #To make it more efficient, it generates sample_count boxes for
        #a GT at once.
        sample_count =IoUs.shape[0]
        sampledBoxSet=self.BoundingBoxGenerator(inputBox.squeeze(), IoUs, sample_count)         

        #Given the generated boxes from a BB, now we map the generated boxes to the image by reshifting and rescaling.
        sampledBoxSet=self.unnormalize(sampledBoxSet, scale[0], shift[0])

        #Clamp the boxes from 0 and imgSize to ensure that they are in the image.
        sampledBoxSet[:,[0,2]]=torch.clamp(sampledBoxSet[:,[0,2]], 0, imgSize[0])
        sampledBoxSet[:,[1,3]]=torch.clamp(sampledBoxSet[:,[1,3]], 0, imgSize[1])

        #Compute the bbox overlaps of the generated boxes.
        generated_box_overlaps=self.computeBoxToBoxIoU(B.expand(sample_count,5)[:,:4], sampledBoxSet).squeeze()

        return sampledBoxSet, generated_box_overlaps

    def sample(self, inputBoxSet, imgSize):
        '''
        INPUTS:
        inputBoxSet     : Input BBs (i.e. ground truths-GTs in Alg.2 in the paper) 
                          Mx5 dimensional tensor. 
                          Each box is represented by [TL_x, TL_y, BR_x, BR_y, gt_label]
        imgSize         : [width, height] of an image   
        '''

        #Normalize the input boxes such that all are shifted/scaled on the reference box
        #that resides at [0.3, 0.3, 0.6, 0.6]. Save scales and shifts, for renormalization 
        #before returning. All operations are conducted within [0, 1] range. Hence we do not,
        #normalize image, we normalize the boxes owing to Theorems 1 and 2.
        inputBoxSet, scales, shifts=self.normalize(inputBoxSet) 

        boxNumber=inputBoxSet.size()[0]

        #Annotations of the datasets may be incorrect especially for small objects.
        #In some cases TL_x=BR_x (same for y). If there is such kind of very rare examples,
        #then we catch the error here, and discard the corrupted annotation.
        validIndices=torch.cuda.ByteTensor(boxNumber).fill_(1)
        flag=0
        for i in range(boxNumber):
              if self.isnan(inputBoxSet[i,0]) or self.isnan(inputBoxSet[i,1]):
                  validIndices[i]=0
                  flag=1
        if flag==1:
              inputBoxSet = inputBoxSet[validIndices,:]  
              scales = scales[validIndices,:]  
              shifts = shifts[validIndices,:]         
              boxNumber=inputBoxSet.size()[0]
        
        # InstanceAllocation determines:
        # 1-perInputAllocation: Number of boxes to be generated for each gt. So, it is a boxNumber sized tensor.
        # 2-positiveRoI_number: In some cases, number of boxes can be 1 or 2 more. So we keep the number of returned boxes.
        #                       The sum of perInputAllocation should also provide this number.
        # 3-inputBoxSetExtended: positiveRoI_numberx5 dimensional array for gts. Basically, each BB in inputBoxSet is 
        #                        duplicated for perInputAllocation[i] times. We use this info to validate/return the IoUs of
        #                        generated boxes on computeBoxToBoxIoU function.
        perInputAllocation, positiveRoI_number, inputBoxSetExtended =self.InstanceAllocation(inputBoxSet)    

        # Another question is the IoU distribution over the boxes. Having estimated the number of generated boxes
        # for each GT, IoUAllocation assigns an IoU using the desired distribution (i.e. self.IoU_weights) for each box.
        IoUSet=self.IoUAllocation(inputBoxSetExtended,positiveRoI_number)  

        #Initialize the necessary data structures to be returned
        sampledBoxSet=torch.cuda.FloatTensor(positiveRoI_number,4).fill_(-1)
        gt_inds=torch.cuda.LongTensor(positiveRoI_number).fill_(0)

        indexPointer=0
        for i in range(boxNumber):
            #BoundingBoxGenerator is doing exactly what Alg.1 in the paper achieves.
            #Given a GT and and IoU, it generates the boxes with the desired IoU.
            #To make it more efficient, it generates perInputAllocation[i] boxes for
            #a GT at once.
            sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:]=self.BoundingBoxGenerator(inputBoxSet[i,:],\
                                                                                              IoUSet[indexPointer:indexPointer+perInputAllocation[i]],\
                                                                                              perInputAllocation[i]) 

            #Given the generated boxes from a GT (also GT), now we map the generated boxes to the image by reshifting and rescaling.
            sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:]=self.unnormalize(sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:], scales[i], shifts[i])
            inputBoxSetExtended[indexPointer:indexPointer+perInputAllocation[i],:4] = self.unnormalize(inputBoxSetExtended[indexPointer:indexPointer+perInputAllocation[i],:4], scales[i], shifts[i])

            #In mmdetection, the association between the boxes are tracked, hence we store the mapping.
            gt_inds[indexPointer:indexPointer+perInputAllocation[i]]=i+1    

            #Update indexpointer to show next empty cell.
            indexPointer+=perInputAllocation[i]

        #Clamp the boxes from 0 and imgSize to ensure that they are in the image.
        sampledBoxSet[:,[0,2]]=torch.clamp(sampledBoxSet[:,[0,2]], 0, imgSize[0])
        sampledBoxSet[:,[1,3]]=torch.clamp(sampledBoxSet[:,[1,3]], 0, imgSize[1])

        #Compute the bbox overlaps of the generated boxes.
        generated_box_overlaps=self.computeBoxToBoxIoU(inputBoxSetExtended[:,:4],sampledBoxSet).squeeze()

        return sampledBoxSet, inputBoxSetExtended[:,-1].type(torch.cuda.LongTensor),generated_box_overlaps,gt_inds

    def normalize(self, boxes):
        #Compute shifts
        shifts = boxes[:,[0,1]]

        #Compute scales
        scales = (torch.cat(((boxes[:,2]-boxes[:,0]).unsqueeze(1), (boxes[:,3]-boxes[:,1]).unsqueeze(1)),1))/(self.reference_box[2]-self.reference_box[0])

        #All the boxes are normalized to reference box. 
        #One can safely following two lines by assigning the boxes[:,:4] to reference box.
        boxes[:,[0,2]]=(boxes[:,[0,2]]-shifts[:,0].unsqueeze(1))/scales[:,0].unsqueeze(1)+self.reference_box[0]
        boxes[:,[1,3]]=(boxes[:,[1,3]]-shifts[:,1].unsqueeze(1))/scales[:,1].unsqueeze(1)+self.reference_box[1]

        return boxes, scales, shifts
  
    def unnormalize(self, boxes,scales,shifts): 
        #self.reference_box[1] will work also, for different reference boxes please correct here.      
        boxes[:,:4]-=self.reference_box[0]

        #Map the normalized boxes to the image coordinates
        boxes[:,[0,2]]=boxes[:,[0,2]]*scales[0]+shifts[0]
        boxes[:,[1,3]]=boxes[:,[1,3]]*scales[1]+shifts[1] 

        return boxes 
  
    def InstanceAllocation(self,inputBoxSet):
        #Determine the number of classes and ensure the sampling to be balanced over classes
        #instead of the instances. Note that this idea originates from OFB sampling in the paper.
        #Here BB generator generates class-balanced examples. Hence determine perClassAllocation
        # in this manner.
        classes=torch.unique(inputBoxSet[:,-1])
        classNumber=classes.size()[0]
        perClassAllocation=math.ceil(self.RoI_number/classNumber)

        #Count the number of instances from each class
        classIndices=torch.cuda.FloatTensor(classNumber,inputBoxSet.size()[0]).fill_(0)
        for i in range(classNumber):
          classIndices[i,:]=inputBoxSet[:,-1]==classes[i]
        classCounts=torch.sum(classIndices,1)  

        #Distribute the perClassAllocation over instances of each class equally
        perInstanceAllocation=torch.ceil(perClassAllocation/classCounts)

        #count the total number of positive examples determined in this fashion
        positiveRoI_number=torch.sum(classCounts*perInstanceAllocation).int()
        extendedInputBoxSet=torch.cuda.FloatTensor(positiveRoI_number,5).fill_(0)
        instanceNumber=inputBoxSet.size()[0]


        indexTracker=0
        perInputAllocation=torch.cuda.FloatTensor(inputBoxSet.size()[0]).fill_(0)


        for i in range(instanceNumber):
          index=classes==inputBoxSet[i,-1]
          extendedInputBoxSet[indexTracker:indexTracker+perInstanceAllocation[index].int()]=inputBoxSet[i,:].expand(perInstanceAllocation[index].int(),5)
          indexTracker+=perInstanceAllocation[index].int()
          perInputAllocation[i]=perInstanceAllocation[index].int()
#        if positiveRoI_number>self.RoI_number:
#            delete_idx=torch.multinomial(perInstanceAllocation,positiveRoI_number-self.RoI_number,replacement=False)
#            pdb.set_trace()          

#            delete_idx=torch.randint(positiveRoI_number, [positiveRoI_number-self.RoI_number])
        return perInputAllocation.int(), positiveRoI_number.item(), extendedInputBoxSet
  
    def IoUAllocation(self,inputBoxSet, positiveRoI_number):
        #Determine the number of examples to be sampled from each bin
        IoUIndices=torch.multinomial(self.IoU_weights,positiveRoI_number,replacement=True)

        #Sample the exact IoUs consdiering the bin length and base of each bin
        IoUSet=(self.IoU_bin_bases[IoUIndices]+torch.rand(positiveRoI_number)*self.bin_width[IoUIndices]).cuda()

        #If IoU is larger than 0.95, then it can be problematic during sampling, so set it to 0.95 for stability.
        IoUSet[IoUSet>0.95]=0.95

        return IoUSet

    def findBottomRightMaxBorders(self,inputBox, IoU, boxArea,proposedx1,proposedy1):
        xA = torch.max(proposedx1, inputBox[0])#alpha
        yA = torch.max(proposedy1, inputBox[1])
        xB = inputBox[2]
        yB = inputBox[3]
        I=torch.clamp(xB - xA,min=0) * torch.clamp(yB - yA,min=0)
        
        limitLeftX=IoU*boxArea+xA*IoU*(inputBox[3]-yA)+xA*(inputBox[3]-yA)-IoU*proposedx1*(inputBox[3]-proposedy1)
        limitLeftX/=((IoU+1)*(inputBox[3]-yA)-IoU*(inputBox[3]-proposedy1))

        limitRightX=(I/IoU-boxArea+I)/(inputBox[3]-proposedy1)
        limitRightX+=proposedx1

        limitTopY=IoU*boxArea+IoU*(inputBox[2]-xA)*yA+yA*(inputBox[2]-xA)-IoU*proposedy1*(inputBox[2]-proposedx1)
        limitTopY/=((IoU+1)*(inputBox[2]-xA)-IoU*(inputBox[2]-proposedx1))

        limitBottomY=(I/IoU-boxArea+I)/(inputBox[2]-proposedx1)
        limitBottomY+=proposedy1
        return limitLeftX,limitRightX,limitTopY,limitBottomY
 
    def findBottomRightBorders(self,inputBox, IoU, boxArea,proposedx1,proposedy1,limitLeftX,limitRightX,limitTopY,limitBottomY):
        xA = torch.max(proposedx1, inputBox[0])#alpha
        yA = torch.max(proposedy1, inputBox[1])
        xB = inputBox[2]
        yB = inputBox[3]
        I=torch.clamp(xB - xA,min=0) * torch.clamp(yB - yA,min=0)
        
        y2TR=torch.arange(limitTopY, inputBox[3]+self.IoU_limit_precision, step=self.IoU_limit_precision).cuda()
        yBnew = torch.min(y2TR, inputBox[3]) 
        Inew=torch.clamp(xB - xA,min=0) * torch.clamp(yBnew - yA,min=0)
        x2TR=(Inew/IoU-boxArea+Inew)/(y2TR-proposedy1)
        x2TR+=proposedx1
        
        x2BR=torch.arange(limitRightX, inputBox[2]-self.IoU_limit_precision, step=-self.IoU_limit_precision).cuda()
        y2BR=(I/IoU-boxArea+I)/(x2BR-proposedx1)
        y2BR+=proposedy1
        
        y2BL=torch.arange(limitBottomY, inputBox[3]-self.IoU_limit_precision, step=-self.IoU_limit_precision).cuda()
        yBnew = torch.min(y2BL, inputBox[3]) 
        x2BL=IoU*boxArea+xA*IoU*(yBnew-yA)+xA*(yBnew-yA)-IoU*proposedx1*(y2BL-proposedy1)
        x2BL/=((IoU+1)*(yBnew-yA)-IoU*(y2BL-proposedy1))    
        
        x2TL=torch.arange(limitLeftX, inputBox[2]+self.IoU_limit_precision, step=self.IoU_limit_precision).cuda()
        xBnew = torch.min(x2TL, inputBox[2]) 
        y2TL=IoU*boxArea+IoU*(xBnew-xA)*yA+yA*(xBnew-xA)-IoU*proposedy1*(x2TL-proposedx1)
        y2TL/=((IoU+1)*(xBnew-xA)-IoU*(x2TL-proposedx1))
        
        x2=torch.cat((x2TR,x2BR,x2BL,x2TL))
        y2=torch.cat((y2TR,y2BR,y2BL,y2TL))
        
        bottomRightBorders=torch.cat((x2.unsqueeze(1),1-y2.unsqueeze(1)),1)
        
        return bottomRightBorders

    def findTopLeftPointBorders(self,inputBox, IoU,boxArea):
        #Top Left
        y1TR=torch.arange((((inputBox[3]*(IoU-1))+ inputBox[1])/IoU), inputBox[1], step=self.IoU_limit_precision).cuda() 
        x1TR=inputBox[2]-(boxArea/(IoU*(inputBox[3]-y1TR)))
        inv_idx = torch.arange(y1TR.size(0)-1, -1, -1).long()
        y1TR = y1TR[inv_idx]
        x1TR = x1TR[inv_idx]
        
        #Top Right
        x1BR=torch.arange(inputBox[0], inputBox[2]-IoU*(inputBox[2]-inputBox[0]), step=self.IoU_limit_precision).cuda() 
        I=(inputBox[2]-x1BR)*(inputBox[3]-inputBox[1])
        y1BR=inputBox[3]-(I/IoU-boxArea+I)/(inputBox[2]-x1BR)

        #Top Left
        y1BL=torch.arange(inputBox[1], inputBox[3]-(boxArea*IoU)/(inputBox[2]-inputBox[0]), step=self.IoU_limit_precision).cuda() 
        x1BL=inputBox[2]-((boxArea*IoU)/((inputBox[3]-y1BL)))

        #Top Right
        y1TL=torch.arange(inputBox[1], inputBox[3]-(boxArea*IoU)/(inputBox[2]-inputBox[0]), step=self.IoU_limit_precision).cuda()
        I=(inputBox[2]-inputBox[0])*(inputBox[3]-y1TL)
        x1TL=inputBox[2]-(I/IoU-boxArea+I)/(inputBox[3]-y1TL)

        inv_idx = torch.arange(y1TL.size(0)-1, -1, -1).long()
        y1TL = y1TL[inv_idx]
        x1TL = x1TL[inv_idx]    
        
        x1=torch.cat((x1TR, x1BR,x1BL,x1TL))
        y1=torch.cat((y1TR, y1BR,y1BL,y1TL))

        P=torch.cat((x1.unsqueeze(1),1-y1.unsqueeze(1)),1)
        
        return P
  
    def BoundingBoxGenerator(self, inputBox, IoUSet, numBoxes):
        sampledBox=torch.cuda.FloatTensor(numBoxes,4).fill_(-1)        
        boxArea=(inputBox[3]-inputBox[1])*(inputBox[2]-inputBox[0])
        box=inputBox
        for i in range(numBoxes):

            #In order to prevent bias for a single corner, decide which corner to pick first
            if np.random.uniform()<0.5:
                flag=1
                inputBox=torch.tensor([1-box[2],1-box[3],1-box[0],1-box[1],box[4]]).cuda() 
            else:
                flag=0
                inputBox=box      

            #Step 1 in Algorithm 1 
            topLeftBorders=self.findTopLeftPointBorders(inputBox, IoUSet[i], boxArea)
            sampledBox[i,0],sampledBox[i,1]=self.samplePolygon(topLeftBorders, inputBox)     

            #Step 2 in Algorithm 1   
            limitLeftX,limitRightX,limitTopY,limitBottomY=self.findBottomRightMaxBorders(inputBox, IoUSet[i], boxArea,sampledBox[i,0],sampledBox[i,1])
            bottomRightBorders=self.findBottomRightBorders(inputBox, IoUSet[i], boxArea, sampledBox[i,0], sampledBox[i,1], limitLeftX, limitRightX, limitTopY, limitBottomY)          
            sampledBox[i,2],sampledBox[i,3]=self.samplePolygon(bottomRightBorders, inputBox)

            #If the box is reversed above then assign the reversed coordinates.
            if flag==1:
                sampledBox[i,:]=torch.tensor([1-sampledBox[i,2],1-sampledBox[i,3],1-sampledBox[i,0],1-sampledBox[i,1]]).cuda()    

        return sampledBox    

    def samplePolygon(self,P, box):
        maxX=torch.max(P[:,0])
        maxY=torch.max(1-P[:,1])
        minX=torch.min(P[:,0])
        minY=torch.min(1-P[:,1])
        inpoly=0
        while inpoly==0:
            proposedx1, proposedy1=self.sampleRectangle([minX,minY,maxX,maxY])
            #Next line is bottleneck  
            p = path.Path(P.cpu().numpy())
            if p.contains_point([proposedx1,1-proposedy1]):
                inpoly=1
        return (proposedx1,proposedy1)    
  
    def sampleRectangle(self,B,numSamples=1):
        x=torch.rand([numSamples])*(B[2]-B[0])+B[0]
        y=torch.rand([numSamples])*(B[3]-B[1])+B[1]
        return (x,y)
    
    def computeBoxToBoxIoU(self,box_a,box_b):
        max_xy = torch.min(box_a[:, 2:].unsqueeze(0), box_b[:, 2:].unsqueeze(0))
        min_xy = torch.max(box_a[:, :2].unsqueeze(0), box_b[:, :2].unsqueeze(0))
        interside = torch.clamp((max_xy - min_xy), min=0)
        inter = interside[:, :, 0] * interside[:, :, 1]       
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(0)  
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0) 
        union = area_a + area_b - inter
        IoU=inter / union
        return IoU
