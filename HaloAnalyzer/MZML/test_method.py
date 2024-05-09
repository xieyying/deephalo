import json
import pymzml
import numpy as np
from tqdm import tqdm
from bintrees import FastAVLTree
import pandas as pd

# TODO: double check this module
class ROI:
    def __init__(self, scan, rt, i, mz, mzmean):
        self.scan = scan
        self.rt = rt
        self.i = i
        self.mz = mz
        self.mzmean = mzmean

    def __repr__(self):
        return 'mz = {:.4f}, rt = {:.2f} - {:.2f}'.format(self.mzmean, self.rt[0], self.rt[1])


class ProcessROI(ROI):
    def __init__(self, scan, rt, i, mz, mzmean):
        super().__init__(scan, rt, i, mz, mzmean)
        self.points = 1

def get_ROIs(path, delta_mz=0.01, required_points=3, gap_points = 8, progress_callback=None,intensity_threshold=10000):
    '''
    :param path: path to mzml file
    :param delta_mz:
    :param required_points:
    :param dropped_points: can be zero points
    :param pbar: an pyQt5 progress bar to visualize
    :return: ROIs - a list of ROI objects found in current file
    '''
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = [scan for scan in run if scan.ms_level == 1]

    ROIs = []  # completed ROIs
    process_ROIs = FastAVLTree()  # processed ROIs

    # initialize a processed data
    number = 1  # number of processed scan
    init_scan = scans[0]
    start_time = init_scan.scan_time[0]

    min_mz = max(init_scan.mz)
    max_mz = min(init_scan.mz)
    for mz, i in zip(init_scan.mz, init_scan.i):
        if i >= intensity_threshold: 
            process_ROIs[mz] = ProcessROI([1, 1],
                                          [start_time, start_time],
                                          [i],
                                          [mz],
                                          mz)
            min_mz = min(min_mz, mz)
            max_mz = max(max_mz, mz)

    for scan in tqdm(scans[1:], desc='Processing scans'):
        number += 1
        process_scan(scan, number, process_ROIs, min_mz, max_mz, delta_mz, intensity_threshold)
        cleanup_process_ROIs(number, process_ROIs, ROIs, required_points)
        min_mz, max_mz = update_min_max_mz(process_ROIs)
        if progress_callback is not None and not number % 10:
            progress_callback.emit(int(number * 100 / len(scans)))

    # check roi 
    for roi in ROIs:
        roi.scan = (roi.scan[0], roi.scan[1])
        assert roi.scan[1] - roi.scan[0] == len(roi.i) - 1

    rois_df = pd.DataFrame()
    for i in range(len(ROIs)):
        d = {'id_roi':i,'mz':ROIs[i].mzmean,'left_base':ROIs[i].scan[0],'right_base':ROIs[i].scan[1]}
        rois_df = pd.concat([rois_df,pd.Series(d)],axis=1)
    
    return rois_df.T


def process_scan(scan, number, process_ROIs, min_mz, max_mz, delta_mz, intensity_threshold):
    for n, mz in enumerate(scan.mz):
        if scan.i[n] >= intensity_threshold:
            ceiling_mz, ceiling_item = None, None
            floor_mz, floor_item = None, None
            if mz < max_mz:
                _, ceiling_item = process_ROIs.ceiling_item(mz)
                ceiling_mz = ceiling_item.mzmean
            if mz > min_mz:
                _, floor_item = process_ROIs.floor_item(mz)
                floor_mz = floor_item.mzmean
            # choose closest
            if ceiling_mz is None and floor_mz is None:
                time = scan.scan_time[0]
                process_ROIs[mz] = ProcessROI([number, number],
                                              [time, time],
                                              [scan.i[n]],
                                              [mz],
                                              mz)
                continue
            elif ceiling_mz is None:
                closest_mz, closest_item = floor_mz, floor_item
            elif floor_mz is None:
                closest_mz, closest_item = ceiling_mz, ceiling_item
            else:
                if ceiling_mz - mz > mz - floor_mz:
                    closest_mz, closest_item = floor_mz, floor_item
                else:
                    closest_mz, closest_item = ceiling_mz, ceiling_item

            if abs(closest_item.mzmean - mz) < delta_mz:
                roi = closest_item
                if roi.scan[1] == number:
                    # ROIs is already extended (two peaks in one mz window)
                    roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                    roi.points += 1
                    roi.mz[-1] = (roi.i[-1]*roi.mz[-1] + scan.i[n]*mz) / (roi.i[-1] + scan.i[n])
                    roi.i[-1] = (roi.i[-1] + scan.i[n])
                
                else:
                    roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                    roi.points += 1
                    roi.mz.append(mz)
                    roi.i.append(scan.i[n])
                    roi.scan[1] = number  # show that we extended the roi
                    roi.rt[1] = scan.scan_time[0]
            else:
                time = scan.scan_time[0]
                process_ROIs[mz] = ProcessROI([number, number],
                                              [time, time],
                                              [scan.i[n]],
                                              [mz],
                                              mz)


def cleanup_process_ROIs(number, process_ROIs, ROIs, required_points):
    to_delete = []
    for mz, roi in process_ROIs.items():
        if roi.scan[1] != number:
            to_delete.append(mz)
            if roi.points >= required_points:
                ROIs.append(ROI(
                    roi.scan,
                    roi.rt,
                    roi.i,
                    roi.mz,
                    roi.mzmean
                ))
    process_ROIs.remove_items(to_delete)


def update_min_max_mz(process_ROIs):
    try:
        min_mz, _ = process_ROIs.min_item()
        max_mz, _ = process_ROIs.max_item()
    except ValueError:
        min_mz = float('inf')
        max_mz = 0
    return min_mz, max_mz

if __name__ == '__main__':
    path = r'J:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\blank\liquid_blank\blank_M10_cmx_pkb_G10.mzML'
    ROIs = get_ROIs(path, delta_mz=0.01, required_points=5, dropped_points=3,intensity_threshold=10000)
    print(len(ROIs))
    print(ROIs)