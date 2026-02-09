FSNet-TF System WorkflowYOLOv5n + Temporal Fusion + Alarm Logic on Edge CPUgraph TD
    %% Global Styles
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef io fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef term fill:#ffebee,stroke:#c62828,stroke-width:2px;

    Start([Start]) --> L1_Select
    class Start term

    %% Layer 1: Input Layer
    subgraph L1 [1. Input Layer]
        L1_Select{Select Input Source}
        L1_OptA[Option A: IP/Mobile Cam<br/>URL: http://ip:8080/video]
        L1_OptB[Option B: Video File<br/>Path: video/testing.mp4]
        L1_Init[OpenCV VideoCapture Init]
        L1_Check{Stream Health Check<br/>isOpened?}
        
        L1_Select -- A --> L1_OptA
        L1_Select -- B --> L1_OptB
        L1_OptA --> L1_Init
        L1_OptB --> L1_Init
        L1_Init --> L1_Check
    end
    
    L1_Check -- No --> Exit_Stream[Exit: Stream Unavailable]
    class Exit_Stream term
    L1_Check -- Yes --> L2_Loop

    %% Layer 2: Frame Control Layer
    subgraph L2 [2. Frame Control Layer]
        L2_Loop(Frame Grab Loop)
        L2_Read[cap.read]
        L2_Valid{Frame Valid?<br/>ret == True}
        L2_Count[frame_idx += 1]
        L2_Gate{Sampling Gate<br/>idx % SKIP == 0}
        L2_Preview[Show Preview Optional]
        
        L2_Loop --> L2_Read
        L2_Read --> L2_Valid
        L2_Valid -- No --> L12_Release
        L2_Valid -- Yes --> L2_Count
        L2_Count --> L2_Gate
        L2_Gate -- No --> L2_Preview
        L2_Preview --> L2_Loop
    end
    
    L2_Gate -- Yes --> L3_Meta

    %% Layer 3: Preprocessing Layer
    subgraph L3 [3. Preprocessing Layer]
        L3_Meta[Read Metadata H,W]
        L3_Resize[Resize to 416x416]
        L3_Color[BGR to RGB]
        L3_Norm[Normalize / 255.0]
        L3_Batch[HWC to CHW + Batch Dim]
        
        L3_Meta --> L3_Resize
        L3_Resize --> L3_Color
        L3_Color --> L3_Norm
        L3_Norm --> L3_Batch
    end

    L3_Batch --> L4_Load

    %% Layer 4: Inference Layer
    subgraph L4 [4. YOLOv5n ONNX Inference]
        L4_Load[Load ONNX Session]
        L4_Run[Forward Pass session.run]
        L4_Parse[Parse Raw Output<br/>xywh, conf, cls]
        L4_Filter[Filter: conf >= Threshold]
        L4_Convert[Convert Norm BBox to Pixel Coords]
        L4_ROI{Valid ROI Check<br/>x2>x1 & y2>y1}
        L4_List[Create Detection List]
        
        L4_Load --> L4_Run
        L4_Run --> L4_Parse
        L4_Parse --> L4_Filter
        L4_Filter --> L4_Convert
        L4_Convert --> L4_ROI
        L4_ROI -- Ignore --> L4_Filter
        L4_ROI -- Keep --> L4_List
    end

    L4_List --> L5_IoU

    %% Layer 5: Tracking Layer
    subgraph L5 [5. Lightweight Tracking]
        L5_IoU[IoU Matching Module]
        L5_Dec{IoU >= Match Thresh?}
        L5_Assign[Assign to Track ID]
        L5_New[Create New Track ID]
        L5_Update[Update History Buffers<br/>bbox, brightness, area]
        
        L5_IoU --> L5_Dec
        L5_Dec -- Yes --> L5_Assign
        L5_Dec -- No --> L5_New
        L5_Assign --> L5_Update
        L5_New --> L5_Update
    end

    L5_Update --> L6_Crop

    %% Layer 6: Feature Extraction
    subgraph L6 [6. Temporal Feature Extraction]
        L6_Crop[Crop ROI from Frame]
        L6_Bright[Calc Brightness bt<br/>Append to History]
        L6_Area[Calc Area At<br/>Append to History]
        
        L6_Crop --> L6_Bright
        L6_Bright --> L6_Area
    end

    L6_Area --> L7_Flicker

    %% Layer 7: Temporal Scoring
    subgraph L7 [7. Temporal Scoring]
        L7_Flicker[Flicker Score Ft<br/>tanh var/100]
        L7_Growth[Growth Score Gt<br/>tanh growth/2]
        
        L7_Flicker --> L7_Growth
    end

    L7_Growth --> L8_Conf

    %% Layer 8: Fusion Layer
    subgraph L8 [8. Fusion Layer]
        L8_Conf[Get YOLO Conf Ct]
        L8_Calc[Calc Safety Score St<br/>0.6Ct + 0.2Ft + 0.2Gt]
        L8_String[Set Track Status String]
        
        L8_Conf --> L8_Calc
        L8_Calc --> L8_String
    end

    L8_String --> L9_Thresh

    %% Layer 9: Alarm Decision
    subgraph L9 [9. Alarm Decision Logic]
        L9_Thresh{St >= Score Thresh?}
        L9_Inc[Streak++]
        L9_Reset[Streak = 0]
        L9_Trig{Streak >= Req AND<br/>Cooldown OK?}
        L9_Act[Action: ALARM + Wav]
        L9_NoAct[Action: Empty]
        
        L9_Thresh -- Yes --> L9_Inc
        L9_Thresh -- No --> L9_Reset
        L9_Inc --> L9_Trig
        L9_Trig -- Yes --> L9_Act
        L9_Trig -- No --> L9_NoAct
        L9_Reset --> L9_NoAct
    end

    L9_Act --> L10_Log
    L9_NoAct --> L10_Log

    %% Layer 10: Logging
    subgraph L10 [10. Logging Layer]
        L10_Log[Append CSV Log Row<br/>Metrics + Action]
    end

    L10_Log --> L11_Draw

    %% Layer 11: Output Layer
    subgraph L11 [11. Output Layer]
        L11_Draw[Draw BBox Red/Green]
        L11_Label[Overlay Label: St, Ft, Gt]
        L11_Show[imshow Frame Display]
        L11_Key{Key == q?}
        
        L11_Draw --> L11_Label
        L11_Label --> L11_Show
        L11_Show --> L11_Key
    end

    L11_Key -- No --> L2_Loop
    L11_Key -- Yes --> L12_Release

    %% Layer 12: Termination
    subgraph L12 [12. Termination]
        L12_Release[Release Capture<br/>Destroy Windows]
        L12_End([END])
        
        L12_Release --> L12_End
    end

    %% Class Assignments
    class L1_Select,L1_Check,L2_Valid,L2_Gate,L4_ROI,L5_Dec,L9_Thresh,L9_Trig,L11_Key decision
    class L1_OptA,L1_OptB,L2_Read,L10_Log,L11_Show,L11_Draw io
    class L5_IoU,L7_Flicker,L7_Growth,L8_Calc,L9_Inc,L9_Reset logic
    class L1_Init,L2_Count,L3_Meta,L3_Resize,L3_Color,L3_Norm,L3_Batch,L4_Load,L4_Run,L4_Parse,L4_Filter,L4_Convert,L4_List,L5_Assign,L5_New,L5_Update,L6_Crop,L6_Bright,L6_Area,L8_Conf,L8_String,L9_Act,L9_NoAct,L12_Release process
