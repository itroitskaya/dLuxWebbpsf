import jax.numpy as np

import poppy
import scipy
from dLux.optics import OpticalLayer

import scipy.special
from dLuxWebbpsf.utils import j1, get_pixel_positions

__all__ = ["InvertY", "InvertX", "InvertXY", "Pad", "Crop", "NircamCirc", \
           "NIRCamFieldAndWavelengthDependentAberration"]

class InvertY(OpticalLayer):
    def __init__(self):
        super().__init__("InvertY")

    def __call__(self, wavefront):
        return wavefront.invert_y()
    
class InvertX(OpticalLayer):
    def __init__(self):
        super().__init__("InvertX")

    def __call__(self, wavefront):
        return wavefront.invert_x()
    
class InvertXY(OpticalLayer):
    def __init__(self):
        super().__init__("InvertXY")

    def __call__(self, wavefront):
        return wavefront.invert_x_and_y()
    
class Pad(OpticalLayer):
    npix_out: int  

    def __init__(self, npix_out):
        self.npix_out = int(npix_out)
        super().__init__("Pad")
    
    def __call__(self, wavefront):
        return wavefront.pad_to(self.npix_out)
    
class Crop(OpticalLayer):
    npix_out: int   

    def __init__(self, npix_out):
        self.npix_out = int(npix_out)
        super().__init__("Crop")
    
    def __call__(self, wavefront):
        # Get relevant parameters
        return wavefront.crop_to(self.npix_out)
    
class NircamCirc(OpticalLayer):
    sigma: float
    diam: None
    npix: None
    oversample: None
    
    def __init__(self, sigma, diam, npix, oversample):
        self.sigma = float(sigma)
        self.diam = float(diam)
        self.npix = int(npix)
        self.oversample = int(oversample)
        
        super().__init__("NircamCirc")
    
    def __call__(self, wavefront):
        
        #jax.debug.print("wavelength: {}", vars(wavefront))
        
        return wavefront.multiply_amplitude(self.get_transmission(wavefront.wavelength, wavefront.pixel_scale))
    
    def get_transmission(self, wavelength, pixelscale):
        #s = 2035
        #jax.debug.print("wavelength start: {}", wavelength)
        #jax.debug.print("pixelscale rad: {}", pixelscale)
        
        pixelscale = pixelscale * 648000.0 / np.pi #from rad to arcsec 
        
        #pixelscale = wavelength * 648000.0 / np.pi
        #pixelscale = pixelscale / self.diam / self.oversample
        
        #jax.debug.print("pixelscale arcsec: {}", pixelscale)
        
        npix = self.npix * self.oversample
        
        x, y = get_pixel_positions((npix, npix), (pixelscale, pixelscale))
        
        #jax.debug.print("x: {}", x[s:-s,s:-s][0].tolist())
        
        r = np.sqrt(x ** 2 + y ** 2)
        #jax.debug.print("r: {}", r[s:-s,s:-s][0].tolist())
        
        sigmar = self.sigma * r
        
        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        bessel_j1_zero2 = scipy.special.jn_zeros(1, 2)[1]
        
        sigmar = sigmar.clip(np.finfo(sigmar.dtype).tiny, bessel_j1_zero2)  # avoid divide by zero -> NaNs
        
        #transmission = (1 - (2 * scipy.special.j1(sigmar) / sigmar) ** 2)
        #transmission = (1 - (2 * jax.scipy.special.bessel_jn(sigmar,v=1)[1] / sigmar) ** 2)
        #transmission = (1 - (2 * jv(1, sigmar) / sigmar) ** 2)
        transmission = (1 - (2 * j1(sigmar) / sigmar) ** 2)
        
        #jax.debug.print("transmission: {}", transmission[s:-s,s:-s][0].tolist())
        #jax.debug.print("transmission: {}", transmission[s:-s,s:-s].tolist())
        
        transmission = np.where(r == 0, 0, transmission)

        # the others have two, one in each corner, both halfway out of the 10x10 box.
        transmission = np.where(
            ((y < -5) & (y > -10)) &
            (np.abs(x) > 7.5) &
            (np.abs(x) < 12.5),
            np.sqrt(1e-3), transmission
        )
        transmission = np.where(
            ((y < 10) & (y > 8)) &
            (np.abs(x) > 9) &
            (np.abs(x) < 11),
            np.sqrt(1e-3), transmission
        )
        
        # mask holder edge
        transmission = np.where(y > 10, 0.0, transmission)

        # edge of mask itself
        # TODO the mask edge is complex and partially opaque based on CV3 images?
        # edge of glass plate rather than opaque mask I believe. To do later.
        # The following is just a temporary placeholder with no quantitative accuracy.
        # but this is outside the coronagraph FOV so that's fine - this only would matter in
        # modeling atypical/nonstandard calibration exposures.
        transmission = np.where((y < -11.5) & (y > -13), 0.7, transmission)
        
        return transmission
    

class NIRCamFieldAndWavelengthDependentAberration(OpticalLayer):
    opd: None
        
    zernike_coeffs: None
    defocus_zern: None
    tilt_zern: None
        
    focusmodel: None
    deltafocus: None
    opd_ref_focus: None
        
    ctilt_model: None
    tilt_offset: None
    tilt_ref_offset: None
    
    def __init__(self, instrument, opd, zernike_coeffs):
        super().__init__("NIRCamFieldAndWavelengthDependentAberration")
        
        self.opd = np.asarray(opd, dtype=float)
        self.zernike_coeffs = np.asarray(zernike_coeffs, dtype=float)

        # Polynomial equations fit to defocus model. Wavelength-dependent focus 
        # results should correspond to Zernike coefficients in meters.
        # Fits were performed to the SW and LW optical design focus model 
        # as provided by Randal Telfer. 
        # See plot at https://github.com/spacetelescope/webbpsf/issues/179
        # The relative wavelength dependence of these focus models are very
        # similar for coronagraphic mode in the Zemax optical prescription,
        # so we opt to use the same focus model in both imaging and coronagraphy.
        defocus_to_rmswfe = -1.09746e7 # convert from mm defocus to meters (WFE)
        sw_focus_cf = np.array([-5.169185169, 50.62919436, -201.5444129, 415.9031962,  
                                -465.9818413, 265.843112, -59.64330811]) / defocus_to_rmswfe
        lw_focus_cf = np.array([0.175718713, -1.100964635, 0.986462016, 1.641692934]) / defocus_to_rmswfe

        # Coronagraphic tilt (`ctilt`) offset model
        # Primarily effects the LW channel (approximately a 0.031mm diff from 3.5um to 5.0um).
        # SW module is small compared to LW, but we include it for completeness.
        # Values have been determined using the Zernike offsets as reported in the 
        # NIRCam Zemax models. The center reference positions will correspond to the 
        # NIRCam target acquisition filters (3.35um for LW and 2.1um for SW)
        sw_ctilt_cf = np.array([125.849834, -289.018704]) / 1e9
        lw_ctilt_cf = np.array([146.827501, -2000.965222, 8385.546158, -11101.658322]) / 1e9

        # Get the representation of focus in the same Zernike basis as used for
        # making the OPD. While it looks like this does more work here than needed
        # by making a whole basis set, in fact because of caching behind the scenes
        # this is actually quick
        basis = poppy.zernike.zernike_basis_faster(
            nterms=len(self.zernike_coeffs),
            npix=self.opd.shape[0],
            outside=0
        )
        self.defocus_zern = np.asarray(basis[3])
        self.tilt_zern = np.asarray(basis[2])

        # Which wavelength was used to generate the OPD map we have already
        # created from zernikes?
        if instrument.channel.upper() == 'SHORT':
            self.focusmodel = sw_focus_cf
            opd_ref_wave = 2.12
        else:
            self.focusmodel = lw_focus_cf
            opd_ref_wave = 3.23
        
        self.opd_ref_focus = np.polyval(self.focusmodel, opd_ref_wave)
        
        print("opd_ref_focus: {}", self.opd_ref_focus)

        # If F323N or F212N, then no focus offset necessary
        if ('F323N' in instrument.filter) or ('F212N' in instrument.filter):
            self.deltafocus = lambda wl: 0
        else:
            self.deltafocus = lambda wl: np.polyval(self.focusmodel, wl) - self.opd_ref_focus

        # Apply wavelength-dependent tilt offset for coronagraphy
        # We want the reference wavelength to be that of the target acq filter
        # Final offset will position TA ref wave at the OPD ref wave location
        #   (wave_um - opd_ref_wave) - (ta_ref_wave - opd_ref_wave) = wave_um - ta_ref_wave
        
        if instrument.channel.upper() == 'SHORT':
            self.ctilt_model = sw_ctilt_cf
            ta_ref_wave = 2.10
        else: 
            self.ctilt_model = lw_ctilt_cf
            ta_ref_wave = 3.35
            
        self.tilt_ref_offset = np.polyval(self.ctilt_model, ta_ref_wave)
        
        print("tilt_ref_offset: {}", self.tilt_ref_offset)

        self.tilt_offset = lambda wl: np.polyval(self.ctilt_model, wl) - self.tilt_ref_offset
        
    def __call__(self, wavefront):
        
        wavelength = wavefront.wavelength * 1e6
        
        #jax.debug.print("wavelength: {}", wavelength)
        
        mod_opd = self.opd - self.deltafocus(wavelength) * self.defocus_zern
        mod_opd = mod_opd + self.tilt_offset(wavelength) * self.tilt_zern
        
        return wavefront.add_opd(mod_opd)


