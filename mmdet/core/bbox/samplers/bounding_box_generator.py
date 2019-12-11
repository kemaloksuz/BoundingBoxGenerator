import torch
import math
import numpy as np
from matplotlib import path
import pdb
class BoxSampler(object):
    def __init__(self, RoINumber):
        super(BoxSampler,self).__init__() 
        self.IoUBases=torch.tensor([0.5,0.6,0.7,0.8,0.9], dtype=torch.float)
        self.RoINumber=RoINumber
        self.IoUDelta=0.1
        self.precision=0.00001

    def isnan(self,x):
        return x != x
    
    def sample(self, inputBoxSet,imgSize, IoUweights):
        inputBoxSet, scales, shifts=self.normalize(inputBoxSet) 
        boxNumber=inputBoxSet.size()[0]
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
        #pdb.set_trace()
        inputBoxSetExtended, positiveRoINumber, perInputAllocation=self.InstanceAllocation(inputBoxSet)    
        IoUSet=self.IoUAllocation(inputBoxSetExtended,positiveRoINumber,IoUweights)  
        sampledBoxSet=torch.cuda.FloatTensor(positiveRoINumber,4).fill_(-1)
        testBoxSet=torch.cuda.FloatTensor(positiveRoINumber,4).fill_(-1)
        gt_inds=torch.cuda.LongTensor(positiveRoINumber).fill_(0)
        indexPointer=0

        for i in range(boxNumber):
            sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:]=self.BoundingBoxGenerator(inputBoxSet[i,:],\
                                                                                              IoUSet[indexPointer:indexPointer+perInputAllocation[i]],\
                                                                                              perInputAllocation[i]) 
            sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:]=self.unnormalize(sampledBoxSet[indexPointer:indexPointer+perInputAllocation[i],:], scales[i], shifts[i])
            testBoxSet[indexPointer:indexPointer+perInputAllocation[i],:]=self.unnormalize(inputBoxSetExtended[indexPointer:indexPointer+perInputAllocation[i],:4], scales[i], shifts[i])
            gt_inds[indexPointer:indexPointer+perInputAllocation[i]]=i+1    
            indexPointer+=perInputAllocation[i]
        sampledBoxSet[:,[0,2]]=torch.clamp(sampledBoxSet[:,[0,2]], 0, imgSize[0])
        sampledBoxSet[:,[1,3]]=torch.clamp(sampledBoxSet[:,[1,3]], 0, imgSize[1])
        generated_box_overlaps=self.computeBoxToBoxIoU(inputBoxSetExtended[:,:4],sampledBoxSet).squeeze()
        #print("IoU",generated_box_overlaps)
        return sampledBoxSet, inputBoxSetExtended[:,-1].type(torch.cuda.LongTensor),generated_box_overlaps,gt_inds

    def normalize(self, boxes):
        shifts = boxes[:,[0,1]]
        scales = (torch.cat(((boxes[:,2]-boxes[:,0]).unsqueeze(1), (boxes[:,3]-boxes[:,1]).unsqueeze(1)),1))/0.3
        boxes[:,[0,2]]=(boxes[:,[0,2]]-shifts[:,0].unsqueeze(1))/scales[:,0].unsqueeze(1)+0.3
        boxes[:,[1,3]]=(boxes[:,[1,3]]-shifts[:,1].unsqueeze(1))/scales[:,1].unsqueeze(1)+0.3
        return boxes, scales, shifts
  
    def unnormalize(self, boxes,scales,shifts):       
        boxes[:,:4]-=0.3
        boxes[:,[0,2]]=boxes[:,[0,2]]*scales[0]+shifts[0]
        boxes[:,[1,3]]=boxes[:,[1,3]]*scales[1]+shifts[1]    
        return boxes 
  
    def InstanceAllocation(self,inputBoxSet):
        classes=torch.unique(inputBoxSet[:,-1])
        classNumber=classes.size()[0]
        perClassAllocation=math.ceil(self.RoINumber/classNumber)
        classIndices=torch.cuda.FloatTensor(classNumber,inputBoxSet.size()[0]).fill_(0)
        for i in range(classNumber):
          classIndices[i,:]=inputBoxSet[:,-1]==classes[i]
        classCounts=torch.sum(classIndices,1)  
        perInstanceAllocation=torch.ceil(perClassAllocation/classCounts)
        positiveRoINumber=torch.sum(classCounts*perInstanceAllocation).int()
        extendedInputBoxSet=torch.cuda.FloatTensor(positiveRoINumber,5).fill_(0)
        instanceNumber=inputBoxSet.size()[0]
        indexTracker=0
        perInputAllocation=torch.cuda.FloatTensor(inputBoxSet.size()[0]).fill_(0)
        for i in range(instanceNumber):
          index=classes==inputBoxSet[i,-1]
          extendedInputBoxSet[indexTracker:indexTracker+perInstanceAllocation[index].int()]=inputBoxSet[i,:].expand(perInstanceAllocation[index].int(),5)
          indexTracker+=perInstanceAllocation[index].int()
          perInputAllocation[i]=perInstanceAllocation[index].int()
        return extendedInputBoxSet, positiveRoINumber.item(), perInputAllocation.int()
  
    def IoUAllocation(self,inputBoxSet, positiveRoINumber,IoUweights):
        IoUIndices=torch.multinomial(IoUweights,positiveRoINumber,replacement=True)
        IoUSet=(self.IoUBases[IoUIndices]+torch.rand(positiveRoINumber)*self.IoUDelta).cuda()
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
        
        y2TR=torch.arange(limitTopY, inputBox[3]+self.precision, step=self.precision).cuda()
        yBnew = torch.min(y2TR, inputBox[3]) 
        Inew=torch.clamp(xB - xA,min=0) * torch.clamp(yBnew - yA,min=0)
        x2TR=(Inew/IoU-boxArea+Inew)/(y2TR-proposedy1)
        x2TR+=proposedx1
        
        x2BR=torch.arange(limitRightX, inputBox[2]-self.precision, step=-self.precision).cuda()
        y2BR=(I/IoU-boxArea+I)/(x2BR-proposedx1)
        y2BR+=proposedy1
        
        y2BL=torch.arange(limitBottomY, inputBox[3]-self.precision, step=-self.precision).cuda()
        yBnew = torch.min(y2BL, inputBox[3]) 
        x2BL=IoU*boxArea+xA*IoU*(yBnew-yA)+xA*(yBnew-yA)-IoU*proposedx1*(y2BL-proposedy1)
        x2BL/=((IoU+1)*(yBnew-yA)-IoU*(y2BL-proposedy1))    
        
        x2TL=torch.arange(limitLeftX, inputBox[2]+self.precision, step=self.precision).cuda()
        xBnew = torch.min(x2TL, inputBox[2]) 
        y2TL=IoU*boxArea+IoU*(xBnew-xA)*yA+yA*(xBnew-xA)-IoU*proposedy1*(x2TL-proposedx1)
        y2TL/=((IoU+1)*(xBnew-xA)-IoU*(x2TL-proposedx1))
        
        x2=torch.cat((x2TR,x2BR,x2BL,x2TL))
        y2=torch.cat((y2TR,y2BR,y2BL,y2TL))
        
        bottomRightBorders=torch.cat((x2.unsqueeze(1),1-y2.unsqueeze(1)),1)
        
        return bottomRightBorders

    def findTopLeftPointBorders(self,inputBox, IoU,boxArea):
        #Top Left
        y1TR=torch.arange((((inputBox[3]*(IoU-1))+ inputBox[1])/IoU), inputBox[1], step=self.precision).cuda() 
        x1TR=inputBox[2]-(boxArea/(IoU*(inputBox[3]-y1TR)))
        inv_idx = torch.arange(y1TR.size(0)-1, -1, -1).long()
        y1TR = y1TR[inv_idx]
        x1TR = x1TR[inv_idx]
        
        #Top Right
        x1BR=torch.arange(inputBox[0], inputBox[2]-IoU*(inputBox[2]-inputBox[0]), step=self.precision).cuda() 
        I=(inputBox[2]-x1BR)*(inputBox[3]-inputBox[1])
        y1BR=inputBox[3]-(I/IoU-boxArea+I)/(inputBox[2]-x1BR)

        #Top Left
        y1BL=torch.arange(inputBox[1], inputBox[3]-(boxArea*IoU)/(inputBox[2]-inputBox[0]), step=self.precision).cuda() 
        x1BL=inputBox[2]-((boxArea*IoU)/((inputBox[3]-y1BL)))

        #Top Right
        y1TL=torch.arange(inputBox[1], inputBox[3]-(boxArea*IoU)/(inputBox[2]-inputBox[0]), step=self.precision).cuda()
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
            topLeftBorders=self.findTopLeftPointBorders(inputBox, IoUSet[i],boxArea)
            sampledBox[i,0],sampledBox[i,1]=self.samplePolygon(topLeftBorders, inputBox)       
            limitLeftX,limitRightX,limitTopY,limitBottomY=self.findBottomRightMaxBorders(inputBox, IoUSet[i], boxArea,sampledBox[i,0],sampledBox[i,1])
            bottomRightBorders=self.findBottomRightBorders(inputBox, IoUSet[i], boxArea, sampledBox[i,0], sampledBox[i,1], limitLeftX, limitRightX, limitTopY, limitBottomY)          
            sampledBox[i,2],sampledBox[i,3]=self.samplePolygon(bottomRightBorders, inputBox)
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
