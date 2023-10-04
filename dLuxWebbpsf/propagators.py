import dLux as dl
import dLux.utils as dlu


__all__ = ["MFT"]


class MFT(dl.MFT):
    """
    Child class of dLux.MFT that changes the output units to arcseconds.
    """

    def __call__(self, wavefront):
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        pixel_scale = dlu.arcsec_to_rad(self.pixel_scale) / self.oversample
        if self.inverse:
            return wavefront.IMFT(
                self.npixels,
                pixel_scale,
                focal_length=self.focal_length,
            )
        else:
            return wavefront.MFT(
                self.npixels,
                pixel_scale,
                focal_length=self.focal_length,
            )
