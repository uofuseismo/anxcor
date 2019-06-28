corr_pack = None


def import_numpy_backend():
    corr_pack = NumpyFFTContract()



class _FFtpack_Contract:


    def __init__(self):
        pass

    def crosscorrelate_two_arrays(self, xarray1, xarray2):
        raise NotImplementedError('please implement crosscorrelate_two_arrays()')


class NumpyFFTContract(_FFtpack_Contract):

    def __init__(self):
        super().__init__()
        import numpy as np
        self.np = np


    def crosscorrelate_two_arrays(self,xarray1, xarray2):
        src_chan_size = xarray1.data.shape[0]
        rec_chan_size = xarray2.data.shape[0]
        time_size     = xarray1.data.shape[-1]
        src_data      = xarray1.data.reshape(src_chan_size, time_size)
        receiver_data = xarray2.data.reshape(rec_chan_size, time_size)

        corr_length = time_size * 2 - 1
        target_length = self.np.fftpack.next_fast_len(corr_length)
        fft_src = self.np.conj(self.np.np.fft.rfft(src_data, target_length, axis=-1))
        fft_rec = self.np.fft.rfft(receiver_data, target_length, axis=-1)

        result = self._multiply_in_mat(fft_src, fft_rec)

        xcorr_mat = self.np.real(
            self.np.fft.fftshift(
            self.np.fft.irfft(result, corr_length, axis=-1), axes=-1)).astype(self.np.float64)
        return xcorr_mat

    def _multiply_in_mat(self,one, two):
        dtype = self.np.complex64
        zero_mat = self.np.zeros((one.shape[0],
                             two.shape[0],
                             one.shape[-1]), dtype=dtype)

        for ri in range(0, two.shape[0]):
            zero_mat[:, ri, :] = one

        for si in range(0, one.shape[0]):
            zero_mat[si, :, :] *= two

        return zero_mat