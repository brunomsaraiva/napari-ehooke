from . import image_manager
from . import segmentation_manager
from . import cell_manager
from magicgui import magic_factory
from napari.types import ImageData, LabelsData


@magic_factory(call_button="Compute Mask",
               algorithm = {"widget_type": "RadioButtons",
                            "orientation": "horizontal",
                            "value": "Isodata",
                            "choices": [("Isodata", "Isodata"), ("Otsu", "Otsu")]},
               img_type = {"widget_type": "RadioButtons",
                           "value": "Phase",
                           "orientation": "horizontal",
                           "choices": [("Phase", "Phase"), ("Fluor", "Fluor")]},
               layout = "vertical",
               auto_call=False)    
def compute_mask(img: ImageData, img_type: str, algorithm: str, mask_closing: int, fill_holes: bool) -> LabelsData:
    return image_manager.compute_binary_mask(img, img_type, algorithm, mask_closing, fill_holes)


@magic_factory(call_button="Align")
def align_img(img: ImageData, mask: LabelsData) -> ImageData:
    return image_manager.align_img_to_mask(img, mask)

@magic_factory(call_button="Segment")
def segment_single_cells(mask: LabelsData) -> LabelsData:
    return segmentation_manager.segment_single_cells(mask)

@magic_factory(call_button="Compute Cells")
def compute_cells(fluor_img: ImageData, mask:LabelsData, labels:LabelsData, pixel_size:int, axial_step:int, find_septum:bool, membrane_thickness:int):
    cellmanager = cell_manager.CellManager()
    cellmanager.compute_cells(fluor_img, mask, labels, pixel_size, axial_step)
    cellmanager.process_cells(fluor_img, mask, find_septum, membrane_thickness)

