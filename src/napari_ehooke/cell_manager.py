import numpy as np
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import threshold_isodata
from skimage.measure import label
from collections import OrderedDict
from napari.utils.notifications import show_info


class Cell(object):

    def __init__(self, label_id):
        self.label = label_id
        self.box = None
        self.box_margin = 5
        self.lines = []
        self.outline = []
        self.neighbours = {}
        self.long_axis = []
        self.short_axis = []

        self.cell_mask = None
        self.memb_mask = None
        self.cyto_mask = None
        self.sept_mask = None

        self.fluor = None
        self.image = None
        self.stats = OrderedDict([("Area", 0),
                                   ("Length", 0),
                                   ("Width", 0),
                                   ("Eccentricity", 0),
                                   ("Irregularity", 0),
                                   ("Baseline", 0),
                                   ("Cell Median", 0),
                                   ("Membrane Median", 0),
                                   ("Cytoplasm Median", 0),
                                   ("Septum Median", 0),
                                   ("Fluor Ratio", 0),
                                   ("Fluor Ratio 25%", 0),
                                   ("Cell Cycle Phase", 0)])

    def add_line(self, y, x1, x2, pixel_size):
        self.lines.append((y, x1, x2))
        self.stats["Area"] = self.stats["Area"] + (x2 - x1 + 1) * float(pixel_size) * float(pixel_size)

    def add_frontier_point(self, x, y, neighs):
        nlabels = []
        notzero = []
        for line in neighs:
            for p in line:
                if p != self.label and not p in nlabels:
                    nlabels.append(p)
                    if p > 0:
                        notzero.append(p)

        if nlabels != []:
            self.outline.append((x, y))

        if notzero != []:
            for l in notzero:
                if l in self.neighbours.keys():
                    count = self.neighbours[l]
                else:
                    count = 0
                self.neighbours[l] = count + 1
    
    def compute_box(self, maskshape):
        points = np.asarray(self.outline)  # in two columns, x, y
        bm = self.box_margin
        w, h = maskshape
        self.box = (max(min(points[:, 0]) - bm, 0),
                    max(min(points[:, 1]) - bm, 0),
                    min(max(points[:, 0]) + bm, w - 1),
                    min(max(points[:, 1]) + bm, h - 1))

    def bound_rectangle(self, points):
        x0, y0 = np.amin(points, axis=0)
        x1, y1 = np.amax(points, axis=0)
        a = np.min([(x1-x0), (y1-y0)])
        return x0, y0, x1, y1, a

    def bounded_value(self, minval, maxval, currval):
        if currval < minval:
            return minval
        elif currval > maxval:
            return maxval
        else:
            return currval

    def bounded_point(self, x0, x1, y0, y1, p):
        tx, ty = p
        tx = self.bounded_value(x0, x1, tx)
        ty = self.bounded_value(y0, y1, ty)
        return tx, ty

    def axes_from_rotation(self, x0, y0, x1, y1, rotation, pixel_size):
        # midpoints
        mx = (x1 + x0) / 2
        my = (y1 + y0) / 2

        # assumes long is X. This duplicates rotations but simplifies
        # using different algorithms such as brightness
        self.long_axis = [[x0, my], [x1, my]]
        self.short_axis = [[mx, y0], [mx, y1]]
        self.short_axis = \
            np.asarray(np.dot(self.short_axis, rotation.T), dtype=np.int32)
        self.long_axis = \
            np.asarray(np.dot(self.long_axis, rotation.T), dtype=np.int32)

        # check if axis fall outside area due to rounding errors
        bx0, by0, bx1, by1 = self.box
        self.short_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, self.short_axis[0])
        self.short_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, self.short_axis[1])
        self.long_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, self.long_axis[0])
        self.long_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, self.long_axis[1])

        self.stats["Length"] = \
            np.linalg.norm(self.long_axis[1] - self.long_axis[0]) * float(pixel_size)
        self.stats["Width"] = \
            np.linalg.norm(self.short_axis[1] - self.short_axis[0]) * float(pixel_size)

    def compute_axes(self, rotations, maskshape, pixel_size):
        self.compute_box(maskshape)
        points = np.asarray(self.outline)  # in two columns, x, y
        width = len(points) + 1

        # no need to do more rotations, due to symmetry
        for rix in range(int(len(rotations) / 2 + 1)):
            r = rotations[rix]
            nx0, ny0, nx1, ny1, nwidth = self.bound_rectangle(
                np.asarray(np.dot(points, r)))

            if nwidth < width:
                width = nwidth
                x0 = nx0
                x1 = nx1
                y0 = ny0
                y1 = ny1
                angle = rix

        self.axes_from_rotation(x0, y0, x1, y1, rotations[angle], pixel_size)

        if self.stats["Length"] < self.stats["Width"]:
            dum = self.stats["Length"]
            self.stats["Length"] = self.stats["Width"]
            self.stats["Width"] = dum
            dum = self.short_axis
            self.short_axis = self.long_axis
            self.long_axis = dum

        self.stats["Eccentricity"] = \
            ((1 - ((self.stats["Width"]/2.0)**2/(self.stats["Length"]/2.0)**2))**0.5)
        self.stats["Irregularity"] = \
            (len(self.outline) * float(pixel_size) / (self.stats["Area"] ** 0.5))

    def fluor_box(self, fluor):
        x0, y0, x1, y1 = self.box
        return fluor[x0:x1+1, y0:y1+1]

    def compute_cell_mask(self):
        x0, y0, x1, y1 = self.box
        mask = np.zeros((x1 - x0 + 1, y1 - y0 + 1))
        for lin in self.lines:
            y, st, en = lin
            mask[st - x0:en - x0 + 1, y - y0] = 1.0
        return mask

    def compute_memb_mask(self, mask, thick):
        eroded = binary_erosion(mask, np.ones((thick * 2 - 1, thick - 1))).astype(float)
        perim = mask - eroded

        return perim

    def compute_sept_mask(self, cell_mask, thick):
        fluor_box = self.fluor
        perim_mask = self.compute_memb_mask(cell_mask, thick)
        inner_mask = cell_mask - perim_mask
        inner_fluor = (inner_mask > 0) * fluor_box

        threshold = threshold_isodata(inner_fluor[inner_fluor > 0])
        interest_matrix = inner_mask * (inner_fluor > threshold)

        label_matrix = label(interest_matrix, connectivity=2)
        interest_label = 0
        interest_label_sum = 0

        for l in range(np.max(label_matrix)):
            if np.sum(img_as_float(label_matrix == l + 1)) > interest_label_sum:
                interest_label = l + 1
                interest_label_sum = np.sum(
                    img_as_float(label_matrix == l + 1))

        return img_as_float(label_matrix == interest_label)

    def recursive_compute_sept(self, cell_mask, inner_mask_thickness):
        try:
            self.sept_mask = self.compute_sept_mask(cell_mask,
                                                    inner_mask_thickness)
        except IndexError:
            try:
                self.recursive_compute_sept(cell_mask, inner_mask_thickness-1)
            except RuntimeError:
                pass

    def compute_regions(self, fluor_image, find_septum, memb_thickness):
        self.fluor = self.fluor_box(fluor_image)
        self.cell_mask = self.compute_cell_mask()

        if find_septum:
            self.recursive_compute_sept(self.cell_mask,
                                        memb_thickness)
            self.perim_mask = self.compute_memb_mask(self.cell_mask, memb_thickness)
                
            self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
            self.cyto_mask = (self.cell_mask - self.perim_mask - self.sept_mask) > 0
        else:
            self.sept_mask = None
            self.memb_mask = self.compute_memb_mask(self.cell_mask, memb_thickness)

    def compute_fluor_baseline(self, mask, fluor):
        margin = 5
        x0, y0, x1, y1 = self.box
        wid, hei = mask.shape
        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, wid - 1)
        y1 = min(y1 + margin, hei - 1)
        mask_box = mask[x0:x1, y0:y1]

        count = 0

        inverted_mask_box = 1 - mask_box

        while count < 5:
            inverted_mask_box = binary_dilation(inverted_mask_box)
            count += 1

        mask_box = 1 - inverted_mask_box

        fluor_box = fluor[x0:x1, y0:y1]
        self.stats["Baseline"] = np.median(
            mask_box[mask_box > 0] * fluor_box[mask_box > 0])

    def measure_fluor(self, fluorbox, roi, fraction=1.0):
        fluorbox = fluorbox
        if roi is not None:
            bright = fluorbox * roi
            bright = bright[roi > 0.5]
            # check if not enough points

            if (len(bright) * fraction) < 1.0:
                return 0.0

            if fraction < 1:
                sortvals = np.sort(bright, axis=None)[::-1]
                sortvals = sortvals[np.nonzero(sortvals)]
                sortvals = sortvals[:int(len(sortvals) * fraction)]
                return np.median(sortvals)

            else:
                return np.median(bright)
        else:
            return 0

    def compute_fluor_stats(self, mask, fluor_img, find_septum):
        self.compute_fluor_baseline(mask, fluor_img)
        fluorbox = self.fluor_box(fluor_img)

        self.stats["Cell Median"] = \
            self.measure_fluor(fluorbox, self.cell_mask) - \
            self.stats["Baseline"]

        self.stats["Membrane Median"] = \
            self.measure_fluor(fluorbox, self.perim_mask) - \
            self.stats["Baseline"]

        self.stats["Cytoplasm Median"] = \
            self.measure_fluor(fluorbox, self.cyto_mask) - \
            self.stats["Baseline"]

        if find_septum:
            self.stats["Septum Median"] = self.measure_fluor(
                fluorbox, self.sept_mask) - self.stats["Baseline"]

            self.stats["Fluor Ratio"] = (self.measure_fluor(fluorbox, self.sept_mask) - self.stats[
                                        "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 75%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.75) - self.stats[
                                            "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 25%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.25) - self.stats[
                                            "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 10%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.10) - self.stats[
                                            "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])
            self.stats["Memb+Sept Median"] = self.measure_fluor(fluorbox, self.membsept_mask) - self.stats["Baseline"]
        else:
            self.stats["Septum Median"] = 0
            self.stats["Fluor Ratio"] = 0
            self.stats["Fluor Ratio 75%"] = 0
            self.stats["Fluor Ratio 25%"] = 0
            self.stats["Fluor Ratio 10%"] = 0
            self.stats["Memb+Sept Median"] = 0

    def set_image(self, find_septum, images, background):
        x0, y0, x1, y1 = self.box
        img = np.zeros((x1 - x0 + 1, (len(images) + 4) * (y1 - y0 + 1)))
        bx0 = 0
        bx1 = x1 - x0 + 1
        by0 = 0
        by1 = y1 - y0 + 1

        for im in images:
            img[bx0:bx1, by0:by1] = im[x0:x1 + 1, y0:y1 + 1]
            by0 = by0 + y1 - y0 + 1
            by1 = by1 + y1 - y0 + 1

        perim = self.perim_mask
        axial = self.sept_mask
        cyto = self.cyto_mask
        img[bx0:bx1, by0:by1] = background[x0:x1 + 1, y0:y1 + 1] * self.cell_mask
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1
        img[bx0:bx1, by0:by1] = background[x0:x1 + 1, y0:y1 + 1] * perim
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1
        img[bx0:bx1, by0:by1] = background[x0:x1 + 1, y0:y1 + 1] * cyto
        if find_septum:
            by0 = by0 + y1 - y0 + 1
            by1 = by1 + y1 - y0 + 1
            img[bx0:bx1, by0:by1] = background[x0:x1 + 1, y0:y1 + 1] * axial
        self.image = img

class CellManager(object):

    def __init__(self):
        self.cells = {}

    def rotation_matrices(self, axial_step):
        result = []
        ang = 0

        while ang < 180:
            sa = np.sin(ang / 180.0 * np.pi)
            ca = np.cos(ang / 180.0 * np.pi)
            # note .T, for column points
            result.append(np.matrix([[ca, -sa], [sa, ca]]).T)
            ang = ang + axial_step

        return result

    def cells_from_labels(self, labels, pixel_size):
        difLabels = []
        for line in labels:
            difLabels.extend(set(line))
        difLabels = sorted(set(difLabels))[1:]

        cells = {}

        for f in difLabels:
            cells[str(int(f))] = Cell(f)

        for y in range(1, len(labels[0, :]) - 1):
            old_label = 0
            x1 = -1
            x2 = -1

            for x in range(1, len(labels[:, 0]) - 1):
                l = int(labels[x, y])

                # check if line began or ended, add line
                if l != old_label:
                    if x1 > 0:
                        x2 = x - 1
                        cells[str(old_label)].add_line(y, x1, x2, pixel_size)
                        x1 = -1
                    if l > 0:
                        x1 = x
                    old_label = l

                # check neighbours
                if l > 0:
                    square = labels[x - 1:x + 2, y - 1:y + 2]
                    cells[str(l)].add_frontier_point(x, y, square)

        for key in cells.keys():
            cells[key].stats["Perimeter"] = len(cells[key].outline) * float(pixel_size)
            cells[key].stats["Neighbours"] = len(cells[key].neighbours)

        self.cells = cells

    def compute_box_axes(self, rotations, maskshape, pixel_size):
        for k in self.cells.keys():
            if self.cells[k].stats["Area"]:
                self.cells[k].compute_axes(rotations, maskshape, pixel_size)

    def compute_cells(self, fluor_img, mask, labels, pixel_size, axial_step):
        self.cells_from_labels(labels, pixel_size)
        rotations = self.rotation_matrices(axial_step)

        self.compute_box_axes(rotations, mask.shape, pixel_size)

    def process_cells(self, fluor_img, mask, find_septum, memb_thickness):
        for k in list(self.cells.keys()):
            try:
                self.cells[k].compute_regions(fluor_img, find_septum, memb_thickness)
                self.cells[k].compute_fluor_stats(mask, fluor_img, find_septum)
            except TypeError:
                del self.cells[k]

        fluorgray = rescale_intensity(img_as_float(fluor_img))
        for k in self.cells.keys():
            self.cells[k].set_image(find_septum, [fluor_img], fluorgray)

        for k in self.cells.keys():
            show_info("Area: " +str(self.cells[k].stats["Area"]) + " FR: " + str(self.cells[k].stats["Fluor Ratio 25%"]))
