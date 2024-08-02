"""EOF analysis for data in `numpy` arrays."""
# (c) Copyright 2000 Jon Saenz, Jesus Fernandez and Juan Zubillaga.
# (c) Copyright 2010-2016 Andrew Dawson. All Rights Reserved.
#
# This file is part of eofs.
#
# eofs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# eofs is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with eofs.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import (absolute_import, division, print_function)  # noqa
import collections
import warnings

import numpy as np
import numpy.ma as ma

import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

try:
    import dask.array
    has_dask = True
except ImportError:
    has_dask = False

from .tools.standard import correlation_map, covariance_map


class Eof(object):
    """EOF analysis (`numpy` interface)"""

    def __init__(self, dataset, weights=None, center=True, ddof=1):
        """Create an Eof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *dataset*
            A `numpy.ndarray`, `numpy.ma.MaskedArray` or `dask.array.Array`
            with two or more dimensions containing the data to be analysed.
            The first dimension is assumed to represent time. Missing
            values are permitted, either in the form of a masked array, or
            `numpy.nan` values. Missing values must be constant with time
            (e.g., values of an oceanographic field over land).

        **Optional arguments:**

        *weights*
            An array of weights whose shape is compatible with those of
            the input array *dataset*. The weights can have the same
            shape as *dataset* or a shape compatible with an array
            broadcast (i.e., the shape of the weights can can match the
            rightmost parts of the shape of the input array *dataset*).
            If the input array *dataset* does not require weighting then
            the value *None* may be used. Defaults to *None* (no
            weighting).

        *center*
            If *True*, the mean along the first axis of *dataset* (the
            time-mean) will be removed prior to analysis. If *False*,
            the mean along the first axis will not be removed. Defaults
            to *True* (mean is removed).

            The covariance interpretation relies on the input data being
            anomaly data with a time-mean of 0. Therefore this option
            should usually be set to *True*. Setting this option to
            *True* has the useful side effect of propagating missing
            values along the time dimension, ensuring that a solution
            can be found even if missing values occur in different
            locations at different times.

        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.

        **Returns:**

        *solver*
            An `Eof` instance.

        **Examples:**

        EOF analysis with no weighting::

            from eofs.standard import Eof
            solver = Eof(data)

        EOF analysis of a data array with spatial dimensions that
        represent latitude and longitude with weighting. In this example
        the data array is dimensioned (ntime, nlat, nlon), and in order
        for the latitude weights to be broadcastable to this shape, an
        extra length-1 dimension is added to the end::

            from eofs.standard import Eof
            import numpy as np
            latitude = np.linspace(-90, 90, 73)
            weights_array = np.cos(np.deg2rad(latitude))[:, np.newaxis]
            solver = Eof(data, weights=weight_array)

        """
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise ValueError('the input data set must be '
                             'at least two dimensional')
        self._data = dataset.copy()
        # Check if the input is a masked array. If so fill it with NaN.
        try:
            self._data = self._data.filled(fill_value=np.nan)
            self._filled = True
        except AttributeError:
            self._filled = False
        # Store information about the shape/size of the input data.
        self._records = self._data.shape[0]
        self._originalshape = self._data.shape[1:]
        channels = np.product(self._originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                # The broadcast_arrays call returns a list, so the second index
                # is retained, but also we want to remove the time dimension
                # from the weights so the the first index from the broadcast
                # array is taken.
                self._weights = np.broadcast_arrays(
                    self._data[0:1], weights)[1][0]
                self._data = self._data * self._weights
            except ValueError:
                raise ValueError('weight array dimensions are incompatible')
            except TypeError:
                raise TypeError('weights are not a valid type')
        else:
            self._weights = None
        # Remove the time mean of the input data unless explicitly told
        # not to by the "center" argument.
        self._centered = center
        if center:
            self._data = self._center(self._data)
        # Reshape to two dimensions (time, space) creating the design matrix.
        self._data = self._data.reshape([self._records, channels])
        # Find the indices of values that are not missing in one row. All the
        # rows will have missing values in the same places provided the
        # array was centered. If it wasn't then it is possible that some
        # missing values will be missed and the singular value decomposition
        # will produce not a number for everything.
        if not self._valid_nan(self._data):
            raise ValueError('missing values detected in different '
                             'locations at different times')
        nonMissingIndex = np.where(np.logical_not(np.isnan(self._data[0])))[0]
        # Remove missing values from the design matrix.
        dataNoMissing = self._data[:, nonMissingIndex]
        if dataNoMissing.size == 0:
            raise ValueError('all input data is missing')
        # Compute the singular value decomposition of the design matrix.
        try:
            if has_dask and isinstance(dataNoMissing, dask.array.Array):
                # Use the parallel Dask algorithm
                dsvd = dask.array.linalg.svd(dataNoMissing)
                A, Lh, E = (x.compute() for x in dsvd)

                # Trim the arrays (since Dask doesn't support
                # 'full_matrices=False')
                A = A[:, :len(Lh)]
                E = E[:len(Lh), :]
            else:
                # Basic numpy algorithm
                A, Lh, E = np.linalg.svd(dataNoMissing, full_matrices=False)

        except (np.linalg.LinAlgError, ValueError):
            raise ValueError('error encountered in SVD, check that missing '
                             'values are in the same places at each time and '
                             'that all the values are not missing')
        # Singular values are the square-root of the eigenvalues of the
        # covariance matrix. Construct the eigenvalues appropriately and
        # normalize by N-ddof where N is the number of observations. This
        # corresponds to the eigenvalues of the normalized covariance matrix.
        self._ddof = ddof
        normfactor = float(self._records - self._ddof)
        # print("Lh: ", Lh)
        # print("normfactor: ", normfactor)
        self._L = Lh * Lh / normfactor
        # print("self._L: ", self._L)
        # Store the number of eigenvalues (and hence EOFs) that were actually
        # computed.
        self.neofs = len(self._L)
        # Re-introduce missing values into the eigenvectors in the same places
        # as they exist in the input maps. Create an array of not-a-numbers
        # and then introduce data values where required. We have to use the
        # astype method to ensure the eigenvectors are the same type as the
        # input dataset since multiplication by np.NaN will promote to 64-bit.
        # print("init eofs")
        self._flatE = np.ones([self.neofs, channels],
                              dtype=self._data.dtype) * np.NaN
        # print("self._flatE_1: ", self._flatE)
        self._flatE = self._flatE.astype(self._data.dtype)
        # print("self._flatE_2: ", self._flatE)
        self._flatE[:, nonMissingIndex] = E
        # print("self._flatE_3: ", self._flatE)
        # Remove the scaling on the principal component time-series that is
        # implicitily introduced by using SVD instead of eigen-decomposition.
        # The PCs may be re-scaled later if required.
        self._P = A * Lh

    def _center(self, in_array):
        """Remove the mean of an array along the first dimension."""
        # Compute the mean along the first dimension.
        mean = in_array.mean(axis=0)
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

    def _valid_nan(self, in_array):
        inan = np.isnan(in_array)
        return (inan.any(axis=0) == inan.all(axis=0)).all()

    def _verify_projection_shape(self, proj_field, proj_space_shape):
        """Verify that a field can be projected onto another"""
        eof_ndim = len(proj_space_shape) + 1
        if eof_ndim - proj_field.ndim not in (0, 1):
            raise ValueError('field has the wrong number of dimensions '
                             'to be projected onto EOFs')
        if proj_field.ndim == eof_ndim:
            check_shape = proj_field.shape[1:]
        else:
            check_shape = proj_field.shape
        if check_shape != proj_space_shape:
            raise ValueError('field has the wrong shape to be projected '
                             ' onto the EOFs')

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled PCs (default).
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of PCs requested is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *pcs*
            An array where the columns are the ordered PCs.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        slicer = slice(0, npcs)
        if pcscaling == 0:
            # Do not scale.
            return self._P[:, slicer].copy()
        elif pcscaling == 1:
            # Divide by the square-root of the eigenvalue.
            return self._P[:, slicer] / np.sqrt(self._L[slicer])
        elif pcscaling == 2:
            # Multiply by the square root of the eigenvalue.
            return self._P[:, slicer] * np.sqrt(self._L[slicer])
        else:
            raise ValueError('invalid PC scaling option: '
                             '{!s}'.format(pcscaling))

    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).

        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EOFs. The following values are
            accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalues.

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs*
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs with no scaling::

            eofs = solver.eofs()

        The leading EOF with scaling applied::

            eof1 = solver.eofs(neofs=1, eofscaling=1)

        """
        if neofs is None or neofs > self.neofs:
            neofs = self.neofs
        slicer = slice(0, neofs)
        neofs = neofs or self.neofs

        print("EOFS_local")
        # self._setEofWH04()

    # CArl Schreck 
        self._flatE[0] = [0.0201374, 0.0212196, 0.0225498, 0.0246027, 0.0260908, 0.0279008, 0.0322317, 0.0346887, 0.0345138, 0.0338656, 0.0303988, 0.0256968, 0.0178561, 0.0102285, 0.00453011, 0.00614469, 0.0132819, 0.0140669, 0.0146942, 0.0156117, 0.0183624, 0.0181316, 0.017729, 0.0181438, 0.0172264, 0.0139292, 0.00930649, 0.00457712, -0.000717312, -0.00457682, -0.00670664, -0.0100101, -0.0137183, -0.0196981, -0.025813, -0.0337611, -0.0408956, -0.0479849, -0.0484854, -0.0447961, -0.0430943, -0.0505088, -0.0535225, -0.0547074, -0.0552229, -0.0579831, -0.0678749, -0.0717353, -0.0703387, -0.0693571, -0.0707544, -0.0735821, -0.0751303, -0.0730513, -0.0715953, -0.0716052, -0.0718432, -0.0699083, -0.0688868, -0.0641975, -0.0597874, -0.0552623, -0.0501976, -0.0436187, -0.0366819, -0.030495, -0.0256779, -0.0178308, -0.0122115, -0.00887276, -0.00600968, -0.00260972, 3.90224e-05, 0.00184028, 0.00384327, 0.00517609, 0.00536145, 0.00619168, 0.00677228, 0.00741259, 0.0078627, 0.00815142, 0.0077056, 0.00556621, 0.00193326, -0.000242119, -0.0016545, -0.00336905, -0.00389634, -0.00299886, -0.00339545, -0.00394282, -0.00437929, -0.0038938, -0.00404911, -0.00272953, -0.00135433, 0.00154235, 0.00386602, 0.00687731, 0.00998862, 0.0127367, 0.0155295, 0.0173478, 0.0188295, 0.0212428, 0.0227043, 0.023628, 0.0245448, 0.0257113, 0.0209384, 0.0176776, 0.0207567, 0.0181543, 0.0192714, 0.0156624, 0.0139781, 0.0154408, 0.0155888, 0.0159456, 0.0160208, 0.016918, 0.0175327, 0.0176317, 0.01743, 0.0184341, 0.0200613, 0.0187954, 0.014405, 0.0120837, 0.0101869, 0.00926868, 0.00937289, 0.0103832, 0.0117766, 0.0134736, 0.0150654, 0.0158359, 0.0160454, 0.0165487, 0.0160455, 0.0167569, 0.0185898, 0.018133, -0.00769157, -0.00698243, -0.00662455, -0.00662111, -0.00664728, -0.00606547, -0.00438712, -0.00190436, 0.000373514, 0.00190309, 0.0036947, 0.00726594, 0.0123041, 0.0166557, 0.0190799, 0.0211441, 0.0251961, 0.0311204, 0.0364453, 0.0397054, 0.0424366, 0.0470069, 0.0535115, 0.0599051, 0.0647376, 0.0684882, 0.0721129, 0.0755184, 0.0782485, 0.0808046, 0.0840393, 0.0876444, 0.0902979, 0.0915534, 0.0922151, 0.0924828, 0.0912462, 0.0882696, 0.0856379, 0.0854632, 0.0865953, 0.0859436, 0.0828683, 0.0799892, 0.0790847, 0.0782669, 0.0748994, 0.0693828, 0.0641287, 0.0597114, 0.0545226, 0.0478804, 0.0413055, 0.0359102, 0.0307764, 0.0246864, 0.0180825, 0.0121473, 0.00687356, 0.0011665, -0.0054465, -0.0123998, -0.0190376, -0.0251422, -0.0307719, -0.0361533, -0.0415277, -0.0468901, -0.0516167, -0.0552519, -0.0580781, -0.0607702, -0.0634341, -0.0658179, -0.0680831, -0.070826, -0.0742776, -0.0780657, -0.0816845, -0.0848312, -0.0872401, -0.0887773, -0.0895861, -0.0899703, -0.0900481, -0.0898451, -0.0895526, -0.0892548, -0.0887724, -0.0880212, -0.0873167, -0.0870046, -0.0868854, -0.0866139, -0.0863483, -0.0864178, -0.0866975, -0.0869166, -0.0872618, -0.0878698, -0.0879444, -0.0865207, -0.0839534, -0.0815094, -0.0794092, -0.0766289, -0.0729821, -0.069806, -0.0679594, -0.0664851, -0.0642112, -0.0613989, -0.0586163, -0.0551112, -0.0500799, -0.0448506, -0.0416956, -0.0409135, -0.0407608, -0.0403254, -0.0404758, -0.0414793, -0.0419738, -0.0411025, -0.0399144, -0.0394365, -0.0386569, -0.0359715, -0.0318139, -0.0281682, -0.0259671, -0.024476, -0.0230017, -0.0217156, -0.020642, -0.0191016, -0.0168365, -0.0145803, -0.0130754, -0.0121898, -0.0113337, -0.0103485, -0.00940524, -0.00853862, -0.01228, -0.0137469, -0.015498, -0.0174376, -0.019051, -0.0200425, -0.0208478, -0.0221674, -0.0240613, -0.0259248, -0.0274498, -0.0290222, -0.0309363, -0.0327655, -0.0340371, -0.0352318, -0.0374099, -0.040811, -0.0445613, -0.0478055, -0.0505966, -0.053383, -0.0561654, -0.0585784, -0.0604649, -0.0619197, -0.063026, -0.0639154, -0.0648608, -0.0659653, -0.0668296, -0.0670171, -0.0666264, -0.0659283, -0.0647346, -0.0628534, -0.061032, -0.0604559, -0.0610782, -0.0614025, -0.0604273, -0.059074, -0.0587639, -0.0592178, -0.0586868, -0.0562655, -0.0528085, -0.0494552, -0.0462145, -0.0424591, -0.0380782, -0.0334394, -0.028729, -0.0239256, -0.0192251, -0.0148561, -0.0105703, -0.00587696, -0.000767209, 0.00427069, 0.00895735, 0.0135548, 0.018365, 0.0232447, 0.0278099, 0.0318479, 0.0354006, 0.0385788, 0.0415054, 0.0443063, 0.0469735, 0.0493299, 0.0512493, 0.0528702, 0.0545275, 0.0564457, 0.0585617, 0.0606407, 0.0624901, 0.0641243, 0.0656639, 0.0670932, 0.0681587, 0.0686882, 0.0688776, 0.0690932, 0.0694483, 0.0697894, 0.0700489, 0.0703865, 0.0708682, 0.0713362, 0.0717237, 0.0722182, 0.0729299, 0.0736012, 0.0739705, 0.0741624, 0.0744402, 0.0747549, 0.074783, 0.0743817, 0.0736134, 0.0724196, 0.0706768, 0.0684931, 0.0660702, 0.0631681, 0.0593634, 0.0549676, 0.0510964, 0.0484687, 0.0464915, 0.0441226, 0.0413382, 0.0390102, 0.0375236, 0.0361749, 0.0343044, 0.0322515, 0.0307502, 0.0298098, 0.0288038, 0.027461, 0.0261657, 0.0252067, 0.0242053, 0.0225907, 0.0202882, 0.0177335, 0.0152805, 0.0129144, 0.0104478, 0.00783532, 0.00518663, 0.0026454, 0.000300452, -0.00183757, -0.00375601, -0.00541928, -0.00683637, -0.00813856, -0.00949339, -0.0108943]
        self._flatE[1] = [0.00958089, 0.0113089, 0.0141104, 0.0142294, 0.011094, 0.00994701, 0.00809211, 0.011731, 0.0124562, 0.0106626, 0.0115963, 0.0132034, 0.0199815, 0.0274937, 0.0304519, 0.0232545, 0.0174511, 0.0159341, 0.0157556, 0.016283, 0.019052, 0.0242762, 0.0311741, 0.0376878, 0.0450025, 0.0514328, 0.0590102, 0.0669728, 0.076423, 0.0857754, 0.0910636, 0.0956418, 0.102474, 0.106402, 0.106235, 0.104352, 0.0989848, 0.0914328, 0.0822559, 0.0687424, 0.0569484, 0.0412172, 0.0367422, 0.0336825, 0.0264891, 0.0138642, -0.00214241, -0.00542421, -0.00721298, -0.00738861, -0.00991884, -0.0144488, -0.0176688, -0.0210441, -0.021692, -0.0248268, -0.0288845, -0.0301265, -0.0334528, -0.0359306, -0.0335921, -0.030334, -0.0318772, -0.0313677, -0.0316697, -0.0306539, -0.0303318, -0.0288865, -0.0291738, -0.0279465, -0.0273986, -0.0270587, -0.0252093, -0.0231686, -0.0217293, -0.0199558, -0.0191789, -0.0173411, -0.017197, -0.015624, -0.0133725, -0.0111757, -0.0106421, -0.0104113, -0.0098338, -0.0101611, -0.00985301, -0.0113636, -0.0114898, -0.0111339, -0.0107688, -0.0100396, -0.00831864, -0.00719017, -0.00766026, -0.008221, -0.00841961, -0.00925039, -0.00909037, -0.00699287, -0.00623772, -0.00492925, -0.00445328, -0.00324854, -0.00228958, -0.00138109, -0.00114546, -0.00128836, -0.00190807, -0.00267862, -0.004161, -0.00371043, 0.00272118, 0.00341484, 0.00192593, -0.00218031, -0.00632215, -0.00898941, -0.00966138, -0.00756295, -0.00682583, -0.00473672, -0.00405507, -0.000858132, 1.08304e-05, -0.00240619, -0.00434712, -0.00581352, -0.00649319, -0.00690525, -0.0088657, -0.00980415, -0.0104891, -0.0107919, -0.0118044, -0.0107099, -0.00947473, -0.00784316, -0.00736479, -0.00641768, -0.0039221, -0.000265673, 0.00396705, 0.00638851, -0.0234452, -0.0237286, -0.0242598, -0.0257541, -0.0278498, -0.0294054, -0.0303809, -0.0319434, -0.03428, -0.0359705, -0.0365111, -0.0376269, -0.0402691, -0.042017, -0.0399291, -0.0351674, -0.0320557, -0.0323623, -0.0334973, -0.0329368, -0.0318463, -0.0325456, -0.0346512, -0.0359585, -0.0357852, -0.0352198, -0.0346471, -0.0331987, -0.0304558, -0.0266977, -0.0217509, -0.0150198, -0.00702443, 0.000885274, 0.00840301, 0.016154, 0.0236249, 0.0294164, 0.0341187, 0.0406856, 0.0503799, 0.060426, 0.0675964, 0.0726156, 0.0786174, 0.0861369, 0.0926036, 0.0963852, 0.0987894, 0.101292, 0.103404, 0.104572, 0.105997, 0.108846, 0.111869, 0.112703, 0.110945, 0.108157, 0.105394, 0.102463, 0.0994906, 0.0974651, 0.0966793, 0.0960792, 0.0948586, 0.0936616, 0.0933611, 0.0935477, 0.0929809, 0.0912897, 0.0889774, 0.0864298, 0.0834015, 0.0798574, 0.0762572, 0.072979, 0.0699297, 0.0668393, 0.0636088, 0.060234, 0.0565644, 0.0525721, 0.0484972, 0.0445474, 0.0406455, 0.0366346, 0.0326256, 0.0288074, 0.0252221, 0.021665, 0.0180298, 0.0143985, 0.0109248, 0.0077149, 0.00484922, 0.00234623, 0.000110995, -0.0019818, -0.00394542, -0.00559749, -0.0067563, -0.00748939, -0.00807443, -0.00863512, -0.00897245, -0.00880599, -0.00824406, -0.00787414, -0.00817657, -0.00886797, -0.00915007, -0.00866971, -0.0079785, -0.0077881, -0.0080362, -0.00815831, -0.0079957, -0.00807151, -0.00875679, -0.0097587, -0.0106593, -0.0115414, -0.0125294, -0.0132587, -0.0132845, -0.0129509, -0.0130156, -0.0136162, -0.014189, -0.0145138, -0.0151912, -0.0166178, -0.0183296, -0.0196762, -0.020636, -0.0214353, -0.0219605, -0.0221473, -0.0224057, -0.023072, -0.0236528, -0.023588, -0.0231723, -0.0231141, 0.0791631, 0.0790005, 0.0788534, 0.0789558, 0.0787191, 0.0774737, 0.0756332, 0.0742641, 0.0736642, 0.073334, 0.0731754, 0.0736516, 0.0745217, 0.0747221, 0.0740168, 0.0737468, 0.0750394, 0.0771997, 0.0786181, 0.0787603, 0.0782321, 0.0774238, 0.0761246, 0.0743177, 0.0723194, 0.0700339, 0.0669886, 0.0631553, 0.0590989, 0.0551053, 0.050795, 0.0457437, 0.0400045, 0.0338417, 0.0274063, 0.0208336, 0.0142543, 0.00766961, 0.00109225, -0.00523726, -0.0112136, -0.0172968, -0.023869, -0.0301866, -0.0349345, -0.0378971, -0.04035, -0.0433452, -0.0464284, -0.048645, -0.0500033, -0.0511831, -0.0523248, -0.0529911, -0.0530197, -0.0526386, -0.0518305, -0.0504277, -0.0487145, -0.0472683, -0.0461335, -0.0447949, -0.0429802, -0.0409365, -0.0388102, -0.0364719, -0.0340076, -0.0318638, -0.0301844, -0.0285471, -0.0265874, -0.0244549, -0.022375, -0.0202376, -0.0179119, -0.015599, -0.0134939, -0.0114276, -0.00913373, -0.00665026, -0.00418182, -0.00177488, 0.000609018, 0.00292501, 0.00518573, 0.00760343, 0.0103284, 0.013287, 0.0164629, 0.0200893, 0.0242832, 0.0287332, 0.0330737, 0.0373579, 0.0418128, 0.046251, 0.0502042, 0.0534924, 0.0563, 0.0588074, 0.0609455, 0.062696, 0.064173, 0.0655986, 0.0672265, 0.0691885, 0.0712013, 0.0726767, 0.0733663, 0.0736165, 0.0736208, 0.0727544, 0.070289, 0.0666258, 0.0630563, 0.0603652, 0.0582434, 0.0562297, 0.0544859, 0.0533236, 0.052608, 0.0520874, 0.0519043, 0.0523358, 0.0533471, 0.0547808, 0.0566722, 0.0590521, 0.0616297, 0.0640005, 0.0660821, 0.0679986, 0.0698162, 0.0715819, 0.0733821, 0.0751242, 0.0764421, 0.0771948, 0.0776564, 0.0780945, 0.0784187, 0.0785314, 0.0786706, 0.0789785]
        
    # Calculated eofs for era
        # self._flatE[0] = [0.025802  ,  0.02697204,  0.02955626,  0.03136224,  0.03025282,0.02953475,  0.03095789,  0.03512703,  0.03617588,  0.0365037 ,0.03469688,  0.03195698,  0.02373563,  0.01643525,  0.00672542,0.00543114,  0.01626796,  0.02039663,  0.0209667 ,  0.02252842,0.02480761,  0.02691578,  0.02782266,  0.02798634,  0.02696193,0.02533598,  0.02218401,  0.01930503,  0.01716662,  0.01532074,0.0144498 ,  0.01145205,  0.0049309 , -0.00125989, -0.0094098 ,-0.01772421, -0.02458174, -0.02995797, -0.02992914, -0.0271112 ,-0.02561346, -0.03495036, -0.03845847, -0.04117322, -0.04238082,-0.04641126, -0.05664212, -0.06117141, -0.05765299, -0.05914931,-0.06400191, -0.06831624, -0.06975077, -0.06573768, -0.06406837,-0.06291663, -0.06110935, -0.05789147, -0.05665942, -0.05348651,-0.05057997, -0.04776391, -0.04251966, -0.03482577, -0.0286124 ,-0.02373866, -0.02046968, -0.01638603, -0.01186216, -0.00772909,-0.00590597, -0.00268391, -0.00016804,  0.00089077,  0.0022605 ,0.0028905 ,  0.0028214 ,  0.0020038 ,  0.00210012,  0.00327198,0.00383779,  0.00445312,  0.00366502,  0.00237631, -0.0002014 ,-0.00365449, -0.00578835, -0.00694433, -0.00724759, -0.00726444,-0.00749744, -0.00839132, -0.00791474, -0.00717338, -0.00720512,-0.00480329, -0.00172824, -0.00016592,  0.00315404,  0.00788308,0.01265726,  0.01576519,  0.01818907,  0.02098885,  0.02290478,0.02497704,  0.02581232,  0.02617146,  0.02680218,  0.02868324,0.02639702,  0.02102636,  0.02462499,  0.0224019 ,  0.01838707,0.0148443 ,  0.0119783 ,  0.01400347,  0.01284905,  0.01348154,0.01219886,  0.0154621 ,  0.01653197,  0.01704034,  0.01714277,0.0166549 ,  0.01747534,  0.01423446,  0.00946654,  0.00807656,0.00665552,  0.00731125,  0.00860434,  0.01037419,  0.01294774,0.01552032,  0.01755845,  0.01936599,  0.02077013,  0.02129303,0.02111529,  0.0227439 ,  0.02493844,  0.02503903, -0.00826012,-0.00842322, -0.00712069, -0.00759732, -0.00573924, -0.00438678,-0.00516471, -0.00462935, -0.00443008, -0.00364963, -0.00333943,0.00014715,  0.00152662,  0.00927601,  0.01479589,  0.01473277,0.02117249,  0.02882283,  0.03163357,  0.03504177,  0.03682052,0.04265477,  0.05010019,  0.05836403,  0.06469107,  0.07073554,0.07413677,  0.07826866,  0.08226398,  0.08560816,  0.08875322,0.09416279,  0.09718248,  0.10131229,  0.10450767,  0.1062184 ,0.10645424,  0.10323697,  0.09840172,  0.09425272,  0.09408065,0.09016395,  0.0875052 ,  0.08504042,  0.08117982,  0.0807711 ,0.08140823,  0.07619787,  0.06961916,  0.06774515,  0.06387148,0.05708054,  0.05113706,  0.04656688,  0.04055468,  0.03477177,0.02599723,  0.0173601 ,  0.01151159,  0.00364262, -0.00463305,-0.01248037, -0.02081398, -0.02766467, -0.03455645, -0.0408871 ,-0.04564734, -0.05148722, -0.05610132, -0.05915112, -0.06179628,-0.06517282, -0.06646573, -0.06829738, -0.07070085, -0.0731478 ,-0.07440899, -0.07669566, -0.07763565, -0.07903771, -0.07934034,-0.07991867, -0.07940933, -0.07884405, -0.07785665, -0.07650015,-0.07551054, -0.07557314, -0.07598098, -0.07526813, -0.07540428,-0.07565872, -0.07554464, -0.07498289, -0.07462723, -0.07513363,-0.07540492, -0.07568218, -0.076492  , -0.07701905, -0.07663594,-0.07524023, -0.07391721, -0.07142707, -0.06909256, -0.06544405,-0.06168765, -0.05428121, -0.05075293, -0.04480915, -0.04368683,-0.03592272, -0.02538908, -0.02901326, -0.02586061, -0.02163527,-0.02647232, -0.02791167, -0.03002471, -0.03396299, -0.03526525,-0.03753329, -0.03828791, -0.03764511, -0.03486495, -0.03398941,-0.0318252 , -0.02964217, -0.02556246, -0.02342696, -0.02107055,-0.01998957, -0.01990735, -0.01951946, -0.01863455, -0.01800798,-0.01735412, -0.01612934, -0.01387106, -0.01325901, -0.01263061,-0.01046811, -0.0090325 , -0.00957175,  0.00732799,  0.00547857,0.00371113,  0.00066749, -0.00245758, -0.00562064, -0.00744888,-0.00853683, -0.01009748, -0.01228563, -0.01534087, -0.01812408,-0.02089047, -0.02356523, -0.02652385, -0.02882632, -0.031532  ,-0.03599697, -0.03985024, -0.04372135, -0.04747998, -0.0517838 ,-0.05639905, -0.06106878, -0.0650982 , -0.06859613, -0.07065784,-0.07272865, -0.0749254 , -0.07596539, -0.07701801, -0.07862605,-0.07842419, -0.07851264, -0.07685689, -0.07422989, -0.07150342,-0.06839607, -0.06578603, -0.06389171, -0.06382096, -0.06199097,-0.06056324, -0.05960597, -0.05862366, -0.05716303, -0.05504734,-0.05150717, -0.04801535, -0.04481192, -0.04161649, -0.03751368,-0.0320879 , -0.0273153 , -0.0219159 , -0.01701085, -0.01209673,-0.00671363, -0.00087101,  0.00540582,  0.01137083,  0.01862352,0.02565659,  0.03264676,  0.03728991,  0.0404342 ,  0.04339146,0.04659244,  0.04973028,  0.05236247,  0.05487868,  0.05741339,0.05879404,  0.06045095,  0.06205555,  0.06326874,  0.0637205 ,0.06499292,  0.06600293,  0.06599351,  0.06644936,  0.066544  ,0.06618861,  0.06576471,  0.06576815,  0.06521787,  0.06567277,0.06596797,  0.0660912 ,  0.0670124 ,  0.0668739 ,  0.06772075,0.06858616,  0.06957171,  0.07112706,  0.07350053,  0.07452578,0.0762085 ,  0.07807966,  0.07980386,  0.08069915,  0.08087409,0.08032698,  0.07996085,  0.07904245,  0.07774054,  0.07650818,0.07533841,  0.07375326,  0.07130406,  0.06752643,  0.0648321 ,0.0622933 ,  0.05751908,  0.05491828,  0.05230077,  0.04927668,0.04827714,  0.04726247,  0.04676419,  0.04670728,  0.04626015,0.04562262,  0.04509737,  0.04389731,  0.04267077,  0.04056577,0.037881  ,  0.03475954,  0.03245365,  0.02937154,  0.02708955,0.02494588,  0.02263749,  0.02083693,  0.01905008,  0.01770884,0.01646873,  0.01541784,  0.01415924,  0.01299725,  0.01095076,0.00938826,  0.00848243]
        # self._flatE[1] = [8.73111554e-03,  1.24783926e-02,  1.63270116e-02,  1.86791281e-02,1.35344103e-02,  1.41204215e-02,  1.25220697e-02,  1.31031353e-02,1.38499364e-02,  1.50525533e-02,  1.57414742e-02,  1.73535853e-02,2.39159524e-02,  3.43536168e-02,  3.24914515e-02,  2.27419291e-02,1.18264165e-02,  9.70376437e-03,  8.00633802e-03,  8.72678884e-03,1.07393566e-02,  1.61498014e-02,  2.20276400e-02,  2.82418360e-02,3.42323389e-02,  4.16141005e-02,  5.06219727e-02,  5.97628058e-02,6.79244697e-02,  7.57700822e-02,  7.95877952e-02,  8.41997413e-02,9.35310660e-02,  9.59684339e-02,  9.67121565e-02,  9.53742829e-02,9.11481173e-02,  8.62149262e-02,  8.05228054e-02,  6.88499906e-02,5.46046997e-02,  3.93044284e-02,  3.36040713e-02,  3.30340060e-02,2.89585587e-02,  1.74702442e-02,  2.74721136e-03, -2.04857270e-03,-3.42712442e-03, -4.97375586e-03, -8.54974053e-03, -1.02427038e-02,-1.29639322e-02, -1.32680220e-02, -1.59974939e-02, -1.61532374e-02,-1.70337227e-02, -2.03310229e-02, -2.34227452e-02, -2.58052322e-02,-2.47562204e-02, -2.32340288e-02, -2.46465819e-02, -2.48186434e-02,-2.42857846e-02, -2.22003625e-02, -2.05935704e-02, -1.80151096e-02,-1.76079746e-02, -1.64258314e-02, -1.63912142e-02, -1.69925414e-02,-1.65853216e-02, -1.31524752e-02, -1.05406757e-02, -8.03152297e-03,-6.43818257e-03, -5.36813265e-03, -4.41028423e-03, -3.42923181e-03,-2.11176956e-03, -4.73229344e-04,  5.96047190e-05, -4.10268229e-04,-1.53596618e-03, -2.17285180e-03, -2.49736714e-03, -3.50148028e-03,-5.02328606e-03, -6.18775892e-03, -6.99553824e-03, -7.75031356e-03,-7.61431786e-03, -7.91827243e-03, -8.88916343e-03, -9.45642341e-03,-1.10140039e-02, -1.19113197e-02, -1.21584435e-02, -1.11733718e-02,-1.03122085e-02, -9.10114722e-03, -7.72385848e-03, -6.12199145e-03,-5.24041721e-03, -5.01404454e-03, -4.90244505e-03, -4.71096513e-03,-4.85704101e-03, -3.59410634e-03, -5.26744910e-04,  3.72276766e-03,1.06476494e-02,  8.75334429e-03, -9.80101917e-04, -6.24138305e-03,-9.67707184e-03, -1.11797284e-02, -1.36971780e-02, -1.26899702e-02,-1.35552717e-02, -1.17801012e-02, -1.09149098e-02, -7.28817095e-03,-7.29018218e-03, -8.37687044e-03, -8.42914343e-03, -8.89271911e-03,-9.29456601e-03, -1.00870829e-02, -9.79597423e-03, -1.04360012e-02,-1.26645851e-02, -1.43943555e-02, -1.45465876e-02, -1.29859632e-02,-1.12609227e-02, -9.65188943e-03, -7.64822135e-03, -6.04173245e-03,-3.27959064e-03, -8.62247442e-04,  2.21133218e-03,  5.16348975e-03,-1.84139804e-02, -1.80609458e-02, -1.83953605e-02, -2.03769381e-02,-2.08861279e-02, -2.28541180e-02, -2.48128315e-02, -2.66913406e-02,-2.80431616e-02, -3.21558751e-02, -3.46257556e-02, -3.35961421e-02,-2.73240045e-02, -3.96833308e-02, -3.42077624e-02, -2.93163322e-02,-2.70460707e-02, -3.36679456e-02, -3.30149898e-02, -3.08755535e-02,-2.97347529e-02, -3.17253939e-02, -3.56108733e-02, -3.85343534e-02,-4.16545671e-02, -4.48868776e-02, -4.69616228e-02, -4.76549911e-02,-4.70176122e-02, -4.44730010e-02, -3.95204174e-02, -3.48392999e-02,-2.99949798e-02, -2.50589873e-02, -1.53618246e-02, -6.78650618e-03,6.56374042e-04,  8.30987740e-03,  1.62146366e-02,  2.54651459e-02,3.87389612e-02,  4.98424598e-02,  5.91787576e-02,  6.22770453e-02,6.93155077e-02,  7.78497284e-02,  8.96644793e-02,  9.23995254e-02,9.33962379e-02,  1.00304856e-01,  1.06950485e-01,  1.06260251e-01,1.07922406e-01,  1.13143620e-01,  1.13455029e-01,  1.16929903e-01,1.16863320e-01,  1.16833146e-01,  1.16945320e-01,  1.12714568e-01,1.09376151e-01,  1.12324485e-01,  1.09318934e-01,  1.06696578e-01,1.04014108e-01,  1.01348421e-01,  9.82623381e-02,  9.58431715e-02,9.34811669e-02,  9.03812254e-02,  8.65506853e-02,  8.16492508e-02,7.77973223e-02,  7.27881104e-02,  6.86831051e-02,  6.48764599e-02,6.17187653e-02,  5.78238525e-02,  5.45096198e-02,  5.02873390e-02,4.70953247e-02,  4.36508197e-02,  4.11115465e-02,  3.74862005e-02,3.43207115e-02,  3.20631145e-02,  3.00530212e-02,  2.79664476e-02,2.56648429e-02,  2.36564789e-02,  2.08652984e-02,  1.73376412e-02,1.48244511e-02,  1.19417883e-02,  8.90022749e-03,  5.75037679e-03,2.85523167e-03,  5.91252486e-04, -1.33663740e-03, -2.99444099e-03,-3.92853364e-03, -4.51645146e-03, -4.08070337e-03, -4.47347831e-03,-4.60949258e-03, -4.21395117e-03, -2.81068430e-03, -2.45816787e-04,-1.76032420e-04,  3.24516546e-04, -1.50272651e-03, -4.56734676e-04,-3.05493042e-04, -3.24673799e-03, -3.85427409e-03, -3.03889414e-03,-4.60656284e-03, -6.40033516e-03, -8.21034344e-03, -7.78770613e-03,-7.81454808e-03, -8.80112053e-03, -1.09946266e-02, -1.18247628e-02,-1.34952765e-02, -1.35648824e-02, -1.33796575e-02, -1.46188029e-02,-1.54026902e-02, -1.64847597e-02, -1.74482892e-02, -1.75358871e-02,-1.76590745e-02, -1.82902793e-02, -1.86521847e-02, -1.87732865e-02,-1.87070598e-02, -1.90397273e-02, -1.94598649e-02, -2.06132732e-02,-2.12333973e-02, -2.06491676e-02, -2.07340916e-02, -1.91609410e-02,8.64153007e-02,  8.54610633e-02,  8.47876521e-02,  8.26009861e-02,8.00813235e-02,  7.76067604e-02,  7.58403994e-02,  7.40885642e-02,7.23258368e-02,  7.19679022e-02,  7.30524358e-02,  7.44269772e-02,7.71629554e-02,  7.75739726e-02,  7.85677594e-02,  8.04002957e-02,8.32188446e-02,  8.56422856e-02,  8.72684792e-02,  8.87103524e-02,8.95242490e-02,  9.02563171e-02,  9.02510426e-02,  8.96870167e-02,8.84025056e-02,  8.62991000e-02,  8.36769803e-02,  7.95560727e-02,7.45345397e-02,  6.93112624e-02,  6.31840210e-02,  5.78283584e-02,5.08084150e-02,  4.22564392e-02,  3.30194353e-02,  2.41084192e-02,1.58640215e-02,  7.44426344e-03, -3.21673055e-04, -8.59574842e-03,-1.64893771e-02, -2.19780976e-02, -2.71117730e-02, -3.28825055e-02,-3.98571645e-02, -4.56445898e-02, -4.82210500e-02, -4.89741987e-02,-5.10593907e-02, -5.25848471e-02, -5.43814502e-02, -5.60773355e-02,-5.72620583e-02, -5.84057073e-02, -6.02623197e-02, -6.04646838e-02,-6.09404173e-02, -6.03927425e-02, -5.99370966e-02, -5.86456589e-02,-5.66554066e-02, -5.48317489e-02, -5.30377508e-02, -5.10277795e-02,-4.88164749e-02, -4.65281998e-02, -4.46792132e-02, -4.27875363e-02,-4.08795672e-02, -3.88026787e-02, -3.67753716e-02, -3.43462028e-02,-3.15017339e-02, -2.88101046e-02, -2.66677160e-02, -2.41864624e-02,-2.21751203e-02, -1.95612302e-02, -1.70562807e-02, -1.38496500e-02,-1.01178177e-02, -7.17956097e-03, -4.63893982e-03, -1.87399056e-03,2.80756005e-04,  1.57211286e-03,  3.73639216e-03,  6.24117622e-03,8.88907592e-03,  1.13133291e-02,  1.38064358e-02,  1.69318683e-02,1.97866785e-02,  2.27679573e-02,  2.64861431e-02,  3.05695563e-02,3.43894472e-02,  3.81322242e-02,  4.20439160e-02,  4.54423744e-02,4.82299280e-02,  5.00624326e-02,  5.15791521e-02,  5.34572192e-02,5.56661640e-02,  5.80675277e-02,  6.00500520e-02,  6.22790030e-02,6.45516289e-02,  6.57616855e-02,  6.60099955e-02,  6.58008175e-02,6.30724132e-02,  5.85996739e-02,  5.41405050e-02,  5.08140153e-02,4.84870580e-02,  4.62426180e-02,  4.51350141e-02,  4.41942059e-02,4.34436675e-02,  4.28012912e-02,  4.32153177e-02,  4.36095626e-02,4.42676920e-02,  4.59151100e-02,  4.83807173e-02,  5.17014892e-02,5.52905193e-02,  5.93100435e-02,  6.30544044e-02,  6.60159904e-02,6.90996442e-02,  7.23694699e-02,  7.55238289e-02,  7.76560482e-02,7.98976157e-02,  8.20430537e-02,  8.41440375e-02,  8.55164331e-02,8.66936336e-02,  8.72609373e-02,  8.76847841e-02,  8.71239331e-02]

        self._L = [55.43719, 52.64146]

#or
        # temp0 = self._flatE[0].copy() # find_eofs.py
        # temp1 = self._flatE[1].copy() # find_eofs.py
        # self._flatE[0] = -1*temp1 # ok for V11 , V12, v13 find_eofs.py 1*temp1
        # self._flatE[1] = temp0 #ok for V11 and V12, v13 find_eofs.py


        # temp0 = self._flatE[0].copy()
        # temp1 = self._flatE[1].copy()
        # self._flatE[0][0:144] = temp0[0:144]
        # self._flatE[1][0:144] = -1*temp1[0:144]
        # self._flatE[0][144:288] = -1*temp0[144:288]
        # self._flatE[1][144:288] = temp1[144:288]
        # self._flatE[0][288:] = -1*temp0[288:]
        # self._flatE[1][288:] = temp1[288:]

        # self._flatE[0] = temp1 # ok
        # self._flatE[1] = -1*temp0 #ok


        # print(repr(self._flatE[1])) 
        # exit()




        flat_eofs = self._flatE[slicer].copy()
        if eofscaling == 0:
            # No modification. A copy needs to be returned in case it is
            # modified. If no copy is made the internally stored eigenvectors
            # could be modified unintentionally.
            rval = flat_eofs
        elif eofscaling == 1:
            # Divide by the square-root of the eigenvalues.
            rval = flat_eofs / np.sqrt(self._L[slicer])[:, np.newaxis]
        elif eofscaling == 2:
            # Multiply by the square-root of the eigenvalues.
            rval = flat_eofs * np.sqrt(self._L[slicer])[:, np.newaxis]
        else:
            raise ValueError('invalid eof scaling option: '
                             '{!s}'.format(eofscaling))
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval.reshape((neofs,) + self._originalshape)

    def eofsAsCorrelation(self, neofs=None):
        """Correlation map EOFs.

        Empirical orthogonal functions (EOFs) expressed as the
        correlation between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        .. note::

            These are not related to the EOFs computed from the
            correlation matrix.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs*
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCorrelation()

        The leading EOF::

            eof1 = solver.eofsAsCorrelation(neofs=1)

        """
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, pcscaling=1)
        # Compute the correlation of the PCs with the input field.
        c = correlation_map(
            pcs,
            self._data.reshape((self._records,) + self._originalshape))
        # The results of the correlation_map function will be a masked array.
        # For consistency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """Covariance map EOFs.

        Empirical orthogonal functions (EOFs) expressed as the
        covariance between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        **Optional arguments:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        *pcscaling*
            Set the scaling of the PCs used to compute covariance. The
            following values are accepted:

            * *0* : Un-scaled PCs.
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue) (default).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

            The default is to divide PCs by the square-root of their
            eigenvalue so that the PCs are scaled to unit variance
            (option 1).

        **Returns:**

        *eofs*
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCovariance()

        The leading EOF::

            eof1 = solver.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs::

            eof1 = solver.eofsAsCovariance(neofs=1, pcscaling=0)

        """
        pcs = self.pcs(npcs=neofs, pcscaling=pcscaling)
        # Divide the input data by the weighting (if any) before computing
        # the covariance maps.
        data = self._data.reshape((self._records,) + self._originalshape)
        if self._weights is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                data /= self._weights
        c = covariance_map(pcs, data, ddof=self._ddof)
        # The results of the covariance_map function will be a masked array.
        # For consitsency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues. If the number of eigenvalues requested is more
            than the number that are available, then all available
            eigenvalues will be returned.

        **Returns:**

        *eigenvalues*
            An array containing the eigenvalues arranged largest to
            smallest.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        # Create a slicer and use it on the eigenvalue array. A copy must be
        # returned in case the slicer takes all elements, in which case a
        # reference to the eigenvalue array is returned. If this is modified
        # then the internal eigenvalues array would then be modified as well.
        slicer = slice(0, neigs)
        return self._L[slicer].copy()

    def varianceFraction(self, neigs=None):
        """Fractional EOF mode variances.

        The fraction of the total variance explained by each EOF mode,
        values between 0 and 1 inclusive.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues. If the number of eigenvalues
            requested is more than the number that are available, then
            fractional variances for all available eigenvalues will be
            returned.

        **Returns:**

        *variance_fractions*
            An array containing the fractional variances.

        **Examples:**

        The fractional variance represented by each EOF mode::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first EOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        # Return the array of eigenvalues divided by the sum of the
        # eigenvalues.
        slicer = slice(0, neigs)
        return self._L[slicer] / self._L.sum()

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).

        **Returns:**

        *total_variance*
            A scalar value.

        **Example:**

        Get the total variance::

            total_variance = solver.totalAnomalyVariance()

        """
        # Return the sum of the eigenvalues.
        return self._L.sum()

    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.

        The method of North et al. (1982) is used to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the result may be inappropriate.

        **Optional arguments:**

        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues. If the
            number of eigenvalues requested is more than the number that
            are available, then typical errors for all available
            eigenvalues will be returned.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `Eof.varianceFraction`. If *False* then no
            scaling is done. Defaults to *False* (no scaling).

        **Returns:**

        *errors*
            An array containing the typical errors.

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Examples:**

        Typical errors for all eigenvalues::

            errors = solver.northTest()

        Typical errors for the first 5 eigenvalues scaled by the sum of
        the eigenvalues::

            errors = solver.northTest(neigs=5, vfscaled=True)

        """
        slicer = slice(0, neigs)
        # Compute the factor that multiplies the eigenvalues. The number of
        # records is assumed to be the number of realizations of the field.
        factor = np.sqrt(2.0 / self._records)
        # If requested, allow for scaling of the eigenvalues by the total
        # variance (sum of the eigenvalues).
        if vfscaled:
            factor /= self._L.sum()
        # Return the typical errors.
        return self._L[slicer] * factor

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the `Eof` instance the returned
        reconstructed field will automatically have this weighting
        removed. Otherwise the returned field will have the same
        weighting as the `Eof` input *dataset*.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be used for the
            reconstruction. Alternatively this argument can be an
            iterable of mode numbers (where the first mode is 1) in
            order to facilitate reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            An array the same shape as the `Eof` input *dataset*
            contaning the reconstruction using *neofs* EOFs.

        **Examples:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        # Determine how the PCs and EOFs will be selected.
        if isinstance(neofs, collectionsAbc.Iterable):
            modes = [m - 1 for m in neofs]
        else:
            modes = slice(0, neofs)
        # Project principal components onto the EOFs to compute the
        # reconstructed field.
        # self._setEofWH04()
        # print("self._flatE[0]: ", self._flatE[0])
        # print("self._flatE[1]: ", self._flatE[1])
        rval = np.dot(self._P[:, modes], self._flatE[modes])
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        rval = rval.reshape((self._records,) + self._originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self._weights is not None:
            rval = rval / self._weights
        # Return the reconstructed field.
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a data set, projects it onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *field*
            A `numpy.ndarray` or `numpy.ma.MaskedArray` with two or more
            dimensions containing the data to be projected onto the
            EOFs. It must have the same corresponding spatial dimensions
            (including missing values in the same places) as the `Eof`
            input *dataset*. *field* may have a different length time
            dimension to the `Eof` input *dataset* or no time dimension
            at all.

        **Optional arguments:**

        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then the field will be projected onto all
            available EOFs.

        *eofscaling*
            Set the scaling of the EOFs that are projected onto. The
            following values are accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then *field* is weighted using the same weights
            used for the EOF analysis prior to projection. If *False*
            then no weighting is applied. Defaults to *True* (weighting
            is applied). Generally only the default setting should be
            used.

        **Returns:**

        *pseudo_pcs*
            An array where the columns are the ordered pseudo-PCs.

        **Examples:**

        Project a data set onto all EOFs::

            pseudo_pcs = solver.projectField(data)

        Project a data set onto the four leading EOFs::

            pseudo_pcs = solver.projectField(data, neofs=4)

        """
        # Check that the shape/dimension of the data set is compatible with
        # the EOFs.
        self._verify_projection_shape(field, self._originalshape)
        input_ndim = field.ndim
        eof_ndim = len(self._originalshape) + 1
        # Create a slice object for truncating the EOFs.
        slicer = slice(0, neofs)
        # If required, weight the data set with the same weighting that was
        # used to compute the EOFs.
        field = field.copy()
        if weighted:
            wts = self.getWeights()
            if wts is not None:
                field = field * wts
        # Fill missing values with NaN if this is a masked array.
        try:
            field = field.filled(fill_value=np.nan)
        except AttributeError:
            pass
        # Flatten the data set into [time, space] dimensionality.
        if eof_ndim > input_ndim:
            field = field.reshape((1,) + field.shape)
        records = field.shape[0]
        channels = np.product(field.shape[1:])
        field_flat = field.reshape([records, channels])
        # Locate the non-missing values and isolate them.
        if not self._valid_nan(field_flat):
            raise ValueError('missing values detected in different '
                             'locations at different times')
        nonMissingIndex = np.where(np.logical_not(np.isnan(field_flat[0])))[0]
        field_flat = field_flat[:, nonMissingIndex]
        # Locate the non-missing values in the EOFs and check they match those
        # in the data set, then isolate the non-missing values.
        eofNonMissingIndex = np.where(
            np.logical_not(np.isnan(self._flatE[0])))[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise ValueError('field and EOFs have different '
                             'missing value locations')
        # print("eofNonMissingIndex: ", eofNonMissingIndex)
        # print("slicer: ", slicer)
        # print("len(self._flatE[0]): ", len(self._flatE[0]))
        # print("self._flatE[0].shape: ", self._flatE[0].shape)
        # self._setEofWH04()
        # print("self._flatE[0]: ", self._flatE[0])
        # print("self._flatE[1]: ", self._flatE[1])
        # temp0 = self._flatE[0].copy()
        # temp1 = self._flatE[1].copy()
        # # temp2 = self._flatE[2].copy()
        # # temp3 = self._flatE[3].copy()
        # # temp4 = self._flatE[4].copy()
        # # temp5 = self._flatE[5].copy()
        # self._flatE[0] = temp1
        # self._flatE[1] = -1*temp0
        # # self._flatE[0] = -1*temp1
        # # self._flatE[1] = -1*temp0
        # # self._flatE[2] = temp3
        # # self._flatE[3] = temp2
        # # self._flatE[4] = temp5
        # self._flatE[5] = temp4

        # self._flatE[1] = -1* self._flatE[1]

        eofs_flat = self._flatE[slicer, eofNonMissingIndex]
        if eofscaling == 1:
            eofs_flat /= np.sqrt(self._L[slicer])[:, np.newaxis]
        elif eofscaling == 2:
            eofs_flat *= np.sqrt(self._L[slicer])[:, np.newaxis]
        # Project the data set onto the EOFs using a matrix multiplication.
        projected_pcs = np.dot(field_flat, eofs_flat.T)

        # print("field_flat shape: ", field_flat.shape, " eofs_flat: ", eofs_flat.shape, "projected_pcs,shape: ", projected_pcs.shape)
        # print("field_flat: ", field_flat)
        # print("field_flat[0][0:432]: ", field_flat[:][0])
        # print("field_flat[[:][:][0].shape: ", field_flat[:][:0].shape)
        # print("field_flat[][0:144]: ", field_flat[0][:144])
        # exit()
        # field_flat[0:144]= field_flat[0:144][::-1]
        # projected_pcs = np.dot(field_flat.T, eofs_flat)
        if eof_ndim > input_ndim:
            # If an extra dimension was introduced, remove it before returning
            # the projected PCs.
            projected_pcs = projected_pcs[0]
        return projected_pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights*
            An array containing the analysis weights.

        **Example:**

        The weights used for the analysis::

            weights = solver.getWeights()

        """
        return self._weights

    def _setEofWH04(self):
        print("_setEofWH04")
        # self._flatE[0] = [0.0201374, 0.0212196, 0.0225498, 0.0246027, 0.0260908, 0.0279008, 0.0322317, 0.0346887, 0.0345138, 0.0338656, 0.0303988, 0.0256968, 0.0178561, 0.0102285, 0.00453011, 0.00614469, 0.0132819, 0.0140669, 0.0146942, 0.0156117, 0.0183624, 0.0181316, 0.017729, 0.0181438, 0.0172264, 0.0139292, 0.00930649, 0.00457712, -0.000717312, -0.00457682, -0.00670664, -0.0100101, -0.0137183, -0.0196981, -0.025813, -0.0337611, -0.0408956, -0.0479849, -0.0484854, -0.0447961, -0.0430943, -0.0505088, -0.0535225, -0.0547074, -0.0552229, -0.0579831, -0.0678749, -0.0717353, -0.0703387, -0.0693571, -0.0707544, -0.0735821, -0.0751303, -0.0730513, -0.0715953, -0.0716052, -0.0718432, -0.0699083, -0.0688868, -0.0641975, -0.0597874, -0.0552623, -0.0501976, -0.0436187, -0.0366819, -0.030495, -0.0256779, -0.0178308, -0.0122115, -0.00887276, -0.00600968, -0.00260972, 3.90224e-05, 0.00184028, 0.00384327, 0.00517609, 0.00536145, 0.00619168, 0.00677228, 0.00741259, 0.0078627, 0.00815142, 0.0077056, 0.00556621, 0.00193326, -0.000242119, -0.0016545, -0.00336905, -0.00389634, -0.00299886, -0.00339545, -0.00394282, -0.00437929, -0.0038938, -0.00404911, -0.00272953, -0.00135433, 0.00154235, 0.00386602, 0.00687731, 0.00998862, 0.0127367, 0.0155295, 0.0173478, 0.0188295, 0.0212428, 0.0227043, 0.023628, 0.0245448, 0.0257113, 0.0209384, 0.0176776, 0.0207567, 0.0181543, 0.0192714, 0.0156624, 0.0139781, 0.0154408, 0.0155888, 0.0159456, 0.0160208, 0.016918, 0.0175327, 0.0176317, 0.01743, 0.0184341, 0.0200613, 0.0187954, 0.014405, 0.0120837, 0.0101869, 0.00926868, 0.00937289, 0.0103832, 0.0117766, 0.0134736, 0.0150654, 0.0158359, 0.0160454, 0.0165487, 0.0160455, 0.0167569, 0.0185898, 0.018133, -0.00769157, -0.00698243, -0.00662455, -0.00662111, -0.00664728, -0.00606547, -0.00438712, -0.00190436, 0.000373514, 0.00190309, 0.0036947, 0.00726594, 0.0123041, 0.0166557, 0.0190799, 0.0211441, 0.0251961, 0.0311204, 0.0364453, 0.0397054, 0.0424366, 0.0470069, 0.0535115, 0.0599051, 0.0647376, 0.0684882, 0.0721129, 0.0755184, 0.0782485, 0.0808046, 0.0840393, 0.0876444, 0.0902979, 0.0915534, 0.0922151, 0.0924828, 0.0912462, 0.0882696, 0.0856379, 0.0854632, 0.0865953, 0.0859436, 0.0828683, 0.0799892, 0.0790847, 0.0782669, 0.0748994, 0.0693828, 0.0641287, 0.0597114, 0.0545226, 0.0478804, 0.0413055, 0.0359102, 0.0307764, 0.0246864, 0.0180825, 0.0121473, 0.00687356, 0.0011665, -0.0054465, -0.0123998, -0.0190376, -0.0251422, -0.0307719, -0.0361533, -0.0415277, -0.0468901, -0.0516167, -0.0552519, -0.0580781, -0.0607702, -0.0634341, -0.0658179, -0.0680831, -0.070826, -0.0742776, -0.0780657, -0.0816845, -0.0848312, -0.0872401, -0.0887773, -0.0895861, -0.0899703, -0.0900481, -0.0898451, -0.0895526, -0.0892548, -0.0887724, -0.0880212, -0.0873167, -0.0870046, -0.0868854, -0.0866139, -0.0863483, -0.0864178, -0.0866975, -0.0869166, -0.0872618, -0.0878698, -0.0879444, -0.0865207, -0.0839534, -0.0815094, -0.0794092, -0.0766289, -0.0729821, -0.069806, -0.0679594, -0.0664851, -0.0642112, -0.0613989, -0.0586163, -0.0551112, -0.0500799, -0.0448506, -0.0416956, -0.0409135, -0.0407608, -0.0403254, -0.0404758, -0.0414793, -0.0419738, -0.0411025, -0.0399144, -0.0394365, -0.0386569, -0.0359715, -0.0318139, -0.0281682, -0.0259671, -0.024476, -0.0230017, -0.0217156, -0.020642, -0.0191016, -0.0168365, -0.0145803, -0.0130754, -0.0121898, -0.0113337, -0.0103485, -0.00940524, -0.00853862, -0.01228, -0.0137469, -0.015498, -0.0174376, -0.019051, -0.0200425, -0.0208478, -0.0221674, -0.0240613, -0.0259248, -0.0274498, -0.0290222, -0.0309363, -0.0327655, -0.0340371, -0.0352318, -0.0374099, -0.040811, -0.0445613, -0.0478055, -0.0505966, -0.053383, -0.0561654, -0.0585784, -0.0604649, -0.0619197, -0.063026, -0.0639154, -0.0648608, -0.0659653, -0.0668296, -0.0670171, -0.0666264, -0.0659283, -0.0647346, -0.0628534, -0.061032, -0.0604559, -0.0610782, -0.0614025, -0.0604273, -0.059074, -0.0587639, -0.0592178, -0.0586868, -0.0562655, -0.0528085, -0.0494552, -0.0462145, -0.0424591, -0.0380782, -0.0334394, -0.028729, -0.0239256, -0.0192251, -0.0148561, -0.0105703, -0.00587696, -0.000767209, 0.00427069, 0.00895735, 0.0135548, 0.018365, 0.0232447, 0.0278099, 0.0318479, 0.0354006, 0.0385788, 0.0415054, 0.0443063, 0.0469735, 0.0493299, 0.0512493, 0.0528702, 0.0545275, 0.0564457, 0.0585617, 0.0606407, 0.0624901, 0.0641243, 0.0656639, 0.0670932, 0.0681587, 0.0686882, 0.0688776, 0.0690932, 0.0694483, 0.0697894, 0.0700489, 0.0703865, 0.0708682, 0.0713362, 0.0717237, 0.0722182, 0.0729299, 0.0736012, 0.0739705, 0.0741624, 0.0744402, 0.0747549, 0.074783, 0.0743817, 0.0736134, 0.0724196, 0.0706768, 0.0684931, 0.0660702, 0.0631681, 0.0593634, 0.0549676, 0.0510964, 0.0484687, 0.0464915, 0.0441226, 0.0413382, 0.0390102, 0.0375236, 0.0361749, 0.0343044, 0.0322515, 0.0307502, 0.0298098, 0.0288038, 0.027461, 0.0261657, 0.0252067, 0.0242053, 0.0225907, 0.0202882, 0.0177335, 0.0152805, 0.0129144, 0.0104478, 0.00783532, 0.00518663, 0.0026454, 0.000300452, -0.00183757, -0.00375601, -0.00541928, -0.00683637, -0.00813856, -0.00949339, -0.0108943]
        # self._flatE[1] = [0.00958089, 0.0113089, 0.0141104, 0.0142294, 0.011094, 0.00994701, 0.00809211, 0.011731, 0.0124562, 0.0106626, 0.0115963, 0.0132034, 0.0199815, 0.0274937, 0.0304519, 0.0232545, 0.0174511, 0.0159341, 0.0157556, 0.016283, 0.019052, 0.0242762, 0.0311741, 0.0376878, 0.0450025, 0.0514328, 0.0590102, 0.0669728, 0.076423, 0.0857754, 0.0910636, 0.0956418, 0.102474, 0.106402, 0.106235, 0.104352, 0.0989848, 0.0914328, 0.0822559, 0.0687424, 0.0569484, 0.0412172, 0.0367422, 0.0336825, 0.0264891, 0.0138642, -0.00214241, -0.00542421, -0.00721298, -0.00738861, -0.00991884, -0.0144488, -0.0176688, -0.0210441, -0.021692, -0.0248268, -0.0288845, -0.0301265, -0.0334528, -0.0359306, -0.0335921, -0.030334, -0.0318772, -0.0313677, -0.0316697, -0.0306539, -0.0303318, -0.0288865, -0.0291738, -0.0279465, -0.0273986, -0.0270587, -0.0252093, -0.0231686, -0.0217293, -0.0199558, -0.0191789, -0.0173411, -0.017197, -0.015624, -0.0133725, -0.0111757, -0.0106421, -0.0104113, -0.0098338, -0.0101611, -0.00985301, -0.0113636, -0.0114898, -0.0111339, -0.0107688, -0.0100396, -0.00831864, -0.00719017, -0.00766026, -0.008221, -0.00841961, -0.00925039, -0.00909037, -0.00699287, -0.00623772, -0.00492925, -0.00445328, -0.00324854, -0.00228958, -0.00138109, -0.00114546, -0.00128836, -0.00190807, -0.00267862, -0.004161, -0.00371043, 0.00272118, 0.00341484, 0.00192593, -0.00218031, -0.00632215, -0.00898941, -0.00966138, -0.00756295, -0.00682583, -0.00473672, -0.00405507, -0.000858132, 1.08304e-05, -0.00240619, -0.00434712, -0.00581352, -0.00649319, -0.00690525, -0.0088657, -0.00980415, -0.0104891, -0.0107919, -0.0118044, -0.0107099, -0.00947473, -0.00784316, -0.00736479, -0.00641768, -0.0039221, -0.000265673, 0.00396705, 0.00638851, -0.0234452, -0.0237286, -0.0242598, -0.0257541, -0.0278498, -0.0294054, -0.0303809, -0.0319434, -0.03428, -0.0359705, -0.0365111, -0.0376269, -0.0402691, -0.042017, -0.0399291, -0.0351674, -0.0320557, -0.0323623, -0.0334973, -0.0329368, -0.0318463, -0.0325456, -0.0346512, -0.0359585, -0.0357852, -0.0352198, -0.0346471, -0.0331987, -0.0304558, -0.0266977, -0.0217509, -0.0150198, -0.00702443, 0.000885274, 0.00840301, 0.016154, 0.0236249, 0.0294164, 0.0341187, 0.0406856, 0.0503799, 0.060426, 0.0675964, 0.0726156, 0.0786174, 0.0861369, 0.0926036, 0.0963852, 0.0987894, 0.101292, 0.103404, 0.104572, 0.105997, 0.108846, 0.111869, 0.112703, 0.110945, 0.108157, 0.105394, 0.102463, 0.0994906, 0.0974651, 0.0966793, 0.0960792, 0.0948586, 0.0936616, 0.0933611, 0.0935477, 0.0929809, 0.0912897, 0.0889774, 0.0864298, 0.0834015, 0.0798574, 0.0762572, 0.072979, 0.0699297, 0.0668393, 0.0636088, 0.060234, 0.0565644, 0.0525721, 0.0484972, 0.0445474, 0.0406455, 0.0366346, 0.0326256, 0.0288074, 0.0252221, 0.021665, 0.0180298, 0.0143985, 0.0109248, 0.0077149, 0.00484922, 0.00234623, 0.000110995, -0.0019818, -0.00394542, -0.00559749, -0.0067563, -0.00748939, -0.00807443, -0.00863512, -0.00897245, -0.00880599, -0.00824406, -0.00787414, -0.00817657, -0.00886797, -0.00915007, -0.00866971, -0.0079785, -0.0077881, -0.0080362, -0.00815831, -0.0079957, -0.00807151, -0.00875679, -0.0097587, -0.0106593, -0.0115414, -0.0125294, -0.0132587, -0.0132845, -0.0129509, -0.0130156, -0.0136162, -0.014189, -0.0145138, -0.0151912, -0.0166178, -0.0183296, -0.0196762, -0.020636, -0.0214353, -0.0219605, -0.0221473, -0.0224057, -0.023072, -0.0236528, -0.023588, -0.0231723, -0.0231141, 0.0791631, 0.0790005, 0.0788534, 0.0789558, 0.0787191, 0.0774737, 0.0756332, 0.0742641, 0.0736642, 0.073334, 0.0731754, 0.0736516, 0.0745217, 0.0747221, 0.0740168, 0.0737468, 0.0750394, 0.0771997, 0.0786181, 0.0787603, 0.0782321, 0.0774238, 0.0761246, 0.0743177, 0.0723194, 0.0700339, 0.0669886, 0.0631553, 0.0590989, 0.0551053, 0.050795, 0.0457437, 0.0400045, 0.0338417, 0.0274063, 0.0208336, 0.0142543, 0.00766961, 0.00109225, -0.00523726, -0.0112136, -0.0172968, -0.023869, -0.0301866, -0.0349345, -0.0378971, -0.04035, -0.0433452, -0.0464284, -0.048645, -0.0500033, -0.0511831, -0.0523248, -0.0529911, -0.0530197, -0.0526386, -0.0518305, -0.0504277, -0.0487145, -0.0472683, -0.0461335, -0.0447949, -0.0429802, -0.0409365, -0.0388102, -0.0364719, -0.0340076, -0.0318638, -0.0301844, -0.0285471, -0.0265874, -0.0244549, -0.022375, -0.0202376, -0.0179119, -0.015599, -0.0134939, -0.0114276, -0.00913373, -0.00665026, -0.00418182, -0.00177488, 0.000609018, 0.00292501, 0.00518573, 0.00760343, 0.0103284, 0.013287, 0.0164629, 0.0200893, 0.0242832, 0.0287332, 0.0330737, 0.0373579, 0.0418128, 0.046251, 0.0502042, 0.0534924, 0.0563, 0.0588074, 0.0609455, 0.062696, 0.064173, 0.0655986, 0.0672265, 0.0691885, 0.0712013, 0.0726767, 0.0733663, 0.0736165, 0.0736208, 0.0727544, 0.070289, 0.0666258, 0.0630563, 0.0603652, 0.0582434, 0.0562297, 0.0544859, 0.0533236, 0.052608, 0.0520874, 0.0519043, 0.0523358, 0.0533471, 0.0547808, 0.0566722, 0.0590521, 0.0616297, 0.0640005, 0.0660821, 0.0679986, 0.0698162, 0.0715819, 0.0733821, 0.0751242, 0.0764421, 0.0771948, 0.0776564, 0.0780945, 0.0784187, 0.0785314, 0.0786706, 0.0789785]
        # self._L = [55.43719, 52.64146]
