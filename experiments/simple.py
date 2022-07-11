import angular

class TestPass:
    pass


class Standard:
    def __init__(self):
        self.neural_text_num_ch = 7
        self.use_shadow_offsets = True
        self.use_offset = True
        self.offset_num_ch = 7

        self.use_offset_2 = False
        self.offset_2_num_ch = 7

        self.shadow_text_num_ch = 7
        self.shadow_offset_num_ch = 7

        self.no_sigma = False

        self.learning_rate = 1e-2
        self.main_angletf = angular.AngularSimple


        self.cosine_factor_out = True

        self.pyramid_sync_point = -1

        self.sigma_start_radius = 8
        self.sigma_1_time = 3600

        self.shadow_mult = 0

        self.nom_min_sigma = 2
        self.nom_scale_down = 2

        self.custom_net = False
        self.custom_net_num_conv = None
        self.custom_net_conv_ch = None




class StandardSigma(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False

class StandardDualOffset(Standard):
    def __init__(self):
        super().__init__()
        self.use_offset_2 = True
        self.no_sigma = False

class StandardSigmaLr3(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3


class StandardRaw(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False

class StandardRawDouble(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False
        self.neural_text_num_ch = 14



class StandardRawLong(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False


        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.shadow_mult = 3


class StandardRawLongRes1(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False


        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.shadow_mult = 3

        self.nom_min_sigma = 1
        self.nom_scale_down = 1


class StandardRawNoShadowLongRes1(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False
        self.use_shadow_offsets = False

        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.nom_min_sigma = 1
        self.nom_scale_down = 1


class StandardRawLongShadowMaskOnly(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False


        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.shadow_mult = 3
        self.use_shadow_offsets = False

        self.nom_min_sigma = 1
        self.nom_scale_down = 1

class StandardRawLShadowNoOffset(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False


        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.shadow_mult = 3
        self.use_shadow_offsets = False

        self.nom_min_sigma = 1
        self.nom_scale_down = 1

        self.use_offset = False

class StandardRawLongShadowMaskOnly1(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False


        self.sigma_start_radius = 8
        self.sigma_1_time = 10000

        self.shadow_mult = 1
        self.use_shadow_offsets = False

        self.nom_min_sigma = 1
        self.nom_scale_down = 1



class StandardRawNoSigmaShadowMaskOnly1(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False

        self.no_sigma = True


        self.shadow_mult = 1
        self.use_shadow_offsets = False

        self.nom_min_sigma = 0
        self.nom_scale_down = 1


class StandardRawNoSigma2(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = True
        self.learning_rate = 1e-3
        self.cosine_factor_out = False

        self.nom_min_sigma = 0
        self.nom_scale_down = 1

class StandardRawNoShadow(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = False
        self.learning_rate = 1e-3
        self.cosine_factor_out = False
        self.use_shadow_offsets = False

class StandardRawNoSigma(Standard):
    def __init__(self):
        super().__init__()
        self.no_sigma = True
        self.learning_rate = 1e-3
        self.cosine_factor_out = False





class StandardRawLongShadowMaskOnly_AS_4_12(StandardRawLongShadowMaskOnly):
    def __init__(self):
        super().__init__()
        self.custom_net = True

        self.custom_net_num_conv = 4
        self.custom_net_conv_ch = 12


class StandardRawLongShadowMaskOnly_AS_4_48(StandardRawLongShadowMaskOnly):
    def __init__(self):
        super().__init__()
        self.custom_net = True

        self.custom_net_num_conv = 4
        self.custom_net_conv_ch = 48



class StandardRawLongShadowMaskOnly_AS(StandardRawLongShadowMaskOnly):
    def __init__(self, custom_net_num_conv, custom_net_conv_ch):
        super().__init__()
        self.custom_net = True

        self.custom_net_num_conv = custom_net_num_conv
        self.custom_net_conv_ch = custom_net_conv_ch

class StandardRawLongShadowMaskOnly_AS_2_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(2, 25)

class StandardRawLongShadowMaskOnly_AS_3_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(3, 25)

class StandardRawLongShadowMaskOnly_AS_5_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(5, 25)

class StandardRawLongShadowMaskOnly_AS_6_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(6, 25)


class StandardRawLongShadowMaskOnly_AS_7_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(7, 25)

class StandardRawLongShadowMaskOnly_AS_8_25(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(8, 25)


class StandardRawLongShadowMaskOnly_AS_4_6(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(4, 6)

class StandardRawLongShadowMaskOnly_AS_4_16(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(4, 16)


class StandardRawLongShadowMaskOnly_AS_4_96(StandardRawLongShadowMaskOnly_AS):
    def __init__(self):
        super().__init__(4, 96)

class VarChannels(StandardRawLongShadowMaskOnly):
    def __init__(self, num_ch):
        super().__init__()

        self.neural_text_num_ch = num_ch
        self.offset_num_ch = num_ch
        self.offset_2_num_ch = num_ch
        self.shadow_text_num_ch = num_ch
        self.shadow_offset_num_ch = num_ch



class VarChannels1(VarChannels):
    def __init__(self):
        super().__init__(1)


class VarChannels3(VarChannels):
    def __init__(self):
        super().__init__(3)


class VarChannels5(VarChannels):
    def __init__(self):
        super().__init__(5)

class VarChannels9(VarChannels):
    def __init__(self):
        super().__init__(9)

class VarChannels11(VarChannels):
    def __init__(self):
        super().__init__(11)


class VarChannels13(VarChannels):
    def __init__(self):
        super().__init__(13)

class VarChannels15(VarChannels):
    def __init__(self):
        super().__init__(15)




class StandardWeb(StandardRawLongShadowMaskOnly):
    def __init__(self):
        super().__init__()
        self.custom_net = True

        self.custom_net_num_conv = 4
        self.custom_net_conv_ch = 24

        self.neural_text_num_ch = 8
        self.offset_num_ch = 6



