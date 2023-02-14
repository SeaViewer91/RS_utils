import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict


def LoadClassDict(
        ClassFile: str
) -> Dict:
    """
    
    :param ClassFile: "Classes.json" 
    :return: Dict
    """
    with open(ClassFile, 'r') as j:
        Classes = json.load(j)

    return Classes


class LabelMe:
    def __init__(self, LabelmeJsonPath: str,
                 ClassDict: Dict):
        """

        :param LabelmeJsonPath: Json file path
        :param ClassDict:
        """
        self.LabelmeJsonPath = LabelmeJsonPath
        self.ClassDict = ClassDict

    def toYOLO(self,
               SaveFile: bool = True,
               SavePath: str = None):
        """

        :param SaveFile:
        :param SavePath:
        :return:
        """
        YOLODict = {
            'classes': [], 'center_x': [], 'center_y': [], 'w': [], 'h': []
        }

        with open(self.LabelmeJsonPath, 'r') as j:
            Annot = json.load(j)

        for shp in Annot['shapes']:
            xmin = shp['points'][0][0]
            ymin = shp['points'][0][1]
            xmax = shp['points'][1][0]
            ymax = shp['points'][1][1]
            center_x = (abs((xmax - xmin) * 0.5) + xmin) / Annot['imageWidth']
            center_y = (abs((ymax - ymin) * 0.5) + ymin) / Annot['imageHeight']
            w = abs(xmax - xmin) / Annot['imageWidth']
            h = abs(ymax - ymin) / Annot['imageHeight']

            if shp['label'] in self.ClassDict.keys():
                ClassID = int(self.ClassDict[shp['label']])

                YOLODict['classes'].append(ClassID)
                YOLODict['center_x'].append(center_x)
                YOLODict['center_y'].append(center_y)
                YOLODict['w'].append(w)
                YOLODict['h'].append(h)

        df = pd.DataFrame(YOLODict)

        if SaveFile:
            df.to_csv(SavePath, index=False, header=None, sep=' ')

        else:
            return df


def main(
        LabelmeDir: str,
        SaveDir: str,
        ClassDict: str
):
    """

    :param LabelmeDir:
    :param SaveDir:
    :param ClassDict:
    :return:
    """
    # ClassDict
    ClassDict = LoadClassDict(ClassDict)

    # labelme Data dir
    ListLabelme = [f for f in os.listdir(LabelmeDir) if f.lower().endswith('.json')]

    # SaveDir
    os.makedirs(SaveDir, exist_ok=True)

    for JsonName in tqdm(ListLabelme):
        Annot = LabelMe(LabelmeJsonPath=os.path.join(LabelmeDir, JsonName), ClassDict=ClassDict)

        # Save
        SaveFileName = os.path.splitext(JsonName)[0] + '.txt'
        Annot.toYOLO(SaveFile=True, SavePath=os.path.join(SaveDir, SaveFileName))


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--LabelmeDir', type=str)
    parser.add_argument('--SaveDir', type=str, help='Save Directory, YOLO Labeling Data')
    parser.add_argument('--ClassDict', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_opt()

    main(
        LabelmeDir=args.LabelmeDir,
        SaveDir=args.SaveDir,
        ClassDict=args.ClassDict
    )
