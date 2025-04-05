from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt


esno_ldpc_32_28 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, ] # 10.0
fer_ldpc_32_28 = [1.0, 1.0, 1.0, 1.0, 0.9803921568627451, 1.0, 0.9803921568627451, 1.0, 1.0, 1.0, 0.9803921568627451, 0.9433962264150944, 0.9803921568627451, 0.9433962264150944, 0.9615384615384616, 0.9615384615384616, 0.9433962264150944, 0.9433962264150944, 0.9090909090909091, 0.9433962264150944, 0.9615384615384616, 0.819672131147541, 0.9433962264150944, 0.9615384615384616, 0.9433962264150944, 0.9433962264150944, 0.9259259259259259, 0.9259259259259259, 0.9090909090909091, 0.8333333333333334, 0.9259259259259259, 0.7936507936507936, 0.7246376811594203, 0.7692307692307693, 0.78125, 0.7692307692307693, 0.7936507936507936, 0.746268656716418, 0.684931506849315, 0.684931506849315, 0.6666666666666666, 0.5555555555555556, 0.5555555555555556, 0.5747126436781609, 0.5376344086021505, 0.5208333333333334, 0.4854368932038835, 0.43478260869565216, 0.43103448275862066, 0.3597122302158273, 0.3816793893129771, 0.4032258064516129, 0.32679738562091504, 0.352112676056338, 0.3125, 0.25, 0.26881720430107525, 0.29069767441860467, 0.25, 0.2183406113537118, 0.19157088122605365, 0.1968503937007874, 0.17301038062283736, 0.1519756838905775, 0.14619883040935672, 0.15625, 0.09823182711198428, 0.09090909090909091, 0.09090909090909091, 0.11210762331838565, 0.076103500761035, 0.0794912559618442, 0.07429420505200594, 0.0657030223390276, 0.03915426781519186, 0.0395882818685669, 0.04892367906066536, 0.03219575016097875, 0.029394473838918283, 0.02399232245681382, 0.019105846388995033, 0.016339869281045753, 0.01924557351809084, 0.016523463317911435, 0.01100594320933304, 0.011834319526627219, 0.008180628272251309, 0.006700616456714018, 0.006544502617801047, 0.006150818058801821, 0.0053803938448294415, 0.004, 0.0037017842600133265, 0.002734631371691096, 0.0025135732957973053, 0.0018253504672897196, 0.0012185908215739318, 0.0009976057462090981, 0.0009575242253629017, 0.000563551728413151, ] # 0.0008907575001781516
ber_ldpc_32_28 = [0.054375, 0.044375, 0.04875, 0.05375, 0.04718137254901961, 0.043125, 0.05024509803921569, 0.0425, 0.04, 0.05875, 0.04779411764705882, 0.04716981132075472, 0.034926470588235295, 0.03773584905660377, 0.046274038461538464, 0.045072115384615384, 0.039504716981132074, 0.04540094339622641, 0.042045454545454546, 0.041273584905660375, 0.05228365384615385, 0.04456967213114754, 0.03714622641509434, 0.043870192307692304, 0.04540094339622641, 0.03891509433962264, 0.059027777777777776, 0.045717592592592594, 0.05568181818181818, 0.04739583333333333, 0.04224537037037037, 0.047619047619047616, 0.04483695652173913, 0.03365384615384615, 0.04931640625, 0.0375, 0.04662698412698413, 0.037779850746268655, 0.04280821917808219, 0.03595890410958904, 0.03916666666666667, 0.03229166666666667, 0.03125, 0.031609195402298854, 0.03528225806451613, 0.02734375, 0.02821601941747573, 0.028532608695652172, 0.02478448275862069, 0.022482014388489208, 0.024093511450381678, 0.028981854838709676, 0.018995098039215685, 0.02222711267605634, 0.019921875, 0.01625, 0.021001344086021504, 0.018168604651162792, 0.0175, 0.012827510917030568, 0.011015325670498084, 0.012549212598425197, 0.01189446366782007, 0.008548632218844984, 0.00941154970760234, 0.011328125, 0.006507858546168959, 0.005965909090909091, 0.0057954545454545455, 0.0075672645739910315, 0.004613774733637747, 0.005216613672496025, 0.004457652303120356, 0.004311760840998686, 0.0023981989036805013, 0.0025237529691211403, 0.0029048434442270057, 0.0019518673535093367, 0.0018738977072310405, 0.0015894913627639156, 0.001253821169277799, 0.0010518790849673202, 0.0011306774441878368, 0.0010017349636483807, 0.0006741140215716487, 0.0007544378698224853, 0.0004908376963350785, 0.00042297641383007235, 0.0003967604712041885, 0.00038058186738836264, 0.0003329118691488217, 0.0002325, 0.0002244206707633079, 0.00016407788230146577, 0.00016024029760707822, 0.00011636609228971963, 7.387706855791962e-05, 6.172685554668795e-05, 5.9845264085181356e-05, 3.416532353504728e-05, ] # 5.177527969785506e-05
plt.plot(esno_ldpc_32_28, fer_ldpc_32_28, label="LDPC(32,28)_1_sp", alpha=1, linewidth=2)
# plt.plot(esno_ldpc_32_28, gaussian_filter1d(fer_ldpc_32_28, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

esno_ldpc_32_28 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0] # 
fer_ldpc_32_28 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.9803921568627451, 1.0, 1.0, 0.9803921568627451, 1.0, 1.0, 1.0, 0.9803921568627451, 1.0, 0.9803921568627451, 0.9803921568627451, 0.9803921568627451, 0.9615384615384616, 0.9433962264150944, 0.9615384615384616, 0.9259259259259259, 1.0, 0.9615384615384616, 0.9433962264150944, 0.9615384615384616, 0.9433962264150944, 0.9433962264150944, 0.847457627118644, 0.9259259259259259, 0.847457627118644, 0.8771929824561403, 0.847457627118644, 0.8333333333333334, 0.7246376811594203, 0.7575757575757576, 0.8064516129032258, 0.819672131147541, 0.5952380952380952, 0.7246376811594203, 0.6756756756756757, 0.6944444444444444, 0.5617977528089888, 0.5882352941176471, 0.5617977528089888, 0.5263157894736842, 0.4672897196261682, 0.5952380952380952, 0.45871559633027525, 0.42016806722689076, 0.4424778761061947, 0.45454545454545453, 0.4, 0.33557046979865773, 0.3048780487804878, 0.25252525252525254, 0.29069767441860467, 0.2183406113537118, 0.21367521367521367, 0.25, 0.1968503937007874, 0.2032520325203252, 0.1557632398753894, 0.16501650165016502, 0.1937984496124031, 0.13440860215053763, 0.13736263736263737, 0.09652509652509653, 0.1026694045174538, 0.08305647840531562, 0.06353240152477764, 0.10245901639344263, 0.07621951219512195, 0.04752851711026616, 0.05636978579481398, 0.04816955684007707, 0.0397456279809221, 0.041084634346754315, 0.028216704288939052, 0.027639579878385848, 0.027609055770292656, 0.017361111111111112, 0.02182453077258839, 0.016113438607798906, 0.015408320493066256, 0.01597954618088846, 0.010850694444444444, 0.007313149041977476, 0.00781616382679381, 0.006243756243756244, 0.007049203440011279, 0.004436950927322744, 0.0033143311679703037, 0.0026733679088916215, 0.0024839783397088777, 0.0017644152727786012, 0.0016607433487228884, 0.0012994100678292056, 0.001400089605734767, 0.0010798669603904799, 0.0006573843989534441, 0.0006381376590558115]
ber_ldpc_32_28 = [0.05, 0.061875, 0.045, 0.0475, 0.0425, 0.0625, 0.044375, 0.04875, 0.05024509803921569, 0.03875, 0.039375, 0.045, 0.05392156862745098, 0.04375, 0.04044117647058824, 0.051470588235294115, 0.03982843137254902, 0.043870192307692304, 0.044221698113207544, 0.040865384615384616, 0.04456018518518518, 0.050625, 0.046875, 0.036556603773584904, 0.043870192307692304, 0.05011792452830189, 0.04658018867924528, 0.034957627118644065, 0.050347222222222224, 0.04184322033898305, 0.04276315789473684, 0.04449152542372881, 0.0421875, 0.030797101449275364, 0.046875, 0.04536290322580645, 0.04252049180327869, 0.03608630952380952, 0.034420289855072464, 0.03969594594594594, 0.044270833333333336, 0.03195224719101124, 0.03272058823529412, 0.03195224719101124, 0.031578947368421054, 0.02394859813084112, 0.03311011904761905, 0.027522935779816515, 0.02100840336134454, 0.02627212389380531, 0.02784090909090909, 0.02175, 0.02139261744966443, 0.018864329268292682, 0.016256313131313132, 0.018531976744186048, 0.015010917030567686, 0.013488247863247864, 0.01625, 0.01218011811023622, 0.013338414634146341, 0.010319314641744548, 0.010829207920792078, 0.012233527131782945, 0.009072580645161291, 0.00987293956043956, 0.0059121621621621625, 0.006416837782340862, 0.005191029900332226, 0.0042487293519695045, 0.006339651639344262, 0.005525914634146341, 0.0029408269961977186, 0.0035583427282976326, 0.0031310211946050095, 0.0026828298887122417, 0.002644823336072309, 0.001904627539503386, 0.0018138474295190713, 0.001708310325786858, 0.0010633680555555555, 0.0014458751636839808, 0.0009970190138575573, 0.0009533898305084746, 0.0009887344199424736, 0.0006781684027777778, 0.00043878894251864854, 0.0004689698296076286, 0.00039023476523476525, 0.000444980967150712, 0.0002662170556393646, 0.00019885987007821823, 0.00016875634924878362, 0.00016145859208107705, 0.00011027595454866258, 9.96446009233733e-05, 8.04009979469321e-05, 8.663054435483871e-05, 6.749168502440499e-05, 4.026479443589845e-05, 3.988360369098822e-05]
# plt.plot(esno_ldpc_32_28, fer_ldpc_32_28, label="LDPC(32,28)_2", alpha=1, linewidth=2)
# plt.plot(esno_ldpc_32_28, gaussian_filter1d(fer_ldpc_32_28, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

esno_gldpc_matrix_by_sp = [2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
fer_gldpc_matrix_by_sp = [0.3546099290780142, 0.35714285714285715, 0.3401360544217687, 0.26455026455026454, 0.21739130434782608, 0.15527950310559005, 0.2358490566037736, 0.11547344110854503, 0.12531328320802004, 0.08620689655172414, 0.1006036217303823, 0.06925207756232687, 0.08503401360544217, 0.04382120946538125, 0.04460303300624442, 0.03692762186115214, 0.03546099290780142, 0.02844141069397042, 0.020366598778004074, 0.019952114924181964, 0.013099292638197537, 0.012254901960784314, 0.008742787200559538, 0.0073551044424830835, 0.005460303592879764, 0.00450815976918222, 0.00406934158053227, 0.002743183189773413, 0.0025040064102564105, 0.0015561295944726277, 0.0015502433882119493]
ber_gldpc_matrix_by_sp = [0.011081560283687944, 0.013169642857142857, 0.012542517006802721, 0.008763227513227513, 0.007201086956521739, 0.005725931677018634, 0.008254716981132075, 0.0040415704387990765, 0.004151002506265664, 0.003556034482758621, 0.003709758551307847, 0.0024238227146814403, 0.002816751700680272, 0.0015337423312883436, 0.0018119982158786797, 0.0016386632200886262, 0.0017065602836879433, 0.0011554323094425483, 0.0008655804480651731, 0.0008853750997605746, 0.0004584752423369138, 0.0004978553921568627, 0.0003059975520195838, 0.00027121947631656367, 0.00021158676422409085, 0.00019159679019024433, 0.00016531700170912346, 0.00011658528556537005, 0.00012050530849358975, 6.613550776508668e-05, 6.007193129321303e-05]
# plt.plot(esno_gldpc_matrix_by_sp, fer_gldpc_matrix_by_sp, label="GLDPC_matrix_decoding_sp", alpha=1, linewidth=2)
# plt.plot(esno_ldpc_32_28, gaussian_filter1d(fer_ldpc_32_28, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

esno_gldpc_32_28_cc_Ham16_11 = [2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
fer_gldpc_32_28_cc_Ham16_11 = [0.6578947368421053, 0.6944444444444444, 0.78125, 0.5555555555555556, 0.6024096385542169, 0.5952380952380952, 0.5319148936170213, 0.44642857142857145, 0.390625, 0.43478260869565216, 0.42016806722689076, 0.3937007874015748, 0.36496350364963503, 0.36496350364963503, 0.23148148148148148, 0.2, 0.22727272727272727, 0.12562814070351758, 0.13736263736263737, 0.12376237623762376, 0.13297872340425532, 0.07429420505200594, 0.052576235541535225, 0.06313131313131314, 0.0395882818685669, 0.03259452411994785, 0.02675227394328518, 0.02147766323024055, 0.018839487565938208, 0.013412017167381975, 0.0070028011204481795]
ber_gldpc_32_28_cc_Ham16_11 = [0.0884046052631579, 0.09505208333333333, 0.1171875, 0.07256944444444445, 0.0677710843373494, 0.07961309523809523, 0.07712765957446809, 0.05552455357142857, 0.049072265625, 0.05326086956521739, 0.048581932773109245, 0.05093503937007874, 0.04698905109489051, 0.045848540145985404, 0.028645833333333332, 0.022, 0.02784090909090909, 0.014761306532663316, 0.015625, 0.014619430693069308, 0.014876994680851064, 0.007661589895988113, 0.005224763406940063, 0.006352588383838384, 0.004280482977038797, 0.0031779661016949155, 0.002959470304975923, 0.0022014604810996563, 0.0021547663903541824, 0.0016597371244635192, 0.0007265406162464986]
plt.plot(esno_gldpc_32_28_cc_Ham16_11, fer_gldpc_32_28_cc_Ham16_11, label="gldpc_32_28_cc_Ham16_11", alpha=1, linewidth=2)
# plt.plot(esno_gldpc_32_28_cc_Ham16_11, gaussian_filter1d(fer_gldpc_32_28_cc_Ham16_11, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

esno_gldpc_32_28_cc_Ham16_11_v2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
fer_gldpc_32_28_cc_Ham16_11_v2 = [0.9090909090909091, 0.9259259259259259, 0.9615384615384616, 0.9259259259259259, 0.8928571428571429, 0.9433962264150944, 0.8771929824561403, 0.9259259259259259, 0.8771929824561403, 0.8928571428571429, 0.9259259259259259, 0.78125, 0.819672131147541, 0.7936507936507936, 0.8620689655172413, 0.6410256410256411, 0.8333333333333334, 0.6756756756756757, 0.704225352112676, 0.7936507936507936, 0.7142857142857143, 0.684931506849315, 0.78125, 0.7142857142857143, 0.5813953488372093, 0.6578947368421053, 0.5882352941176471, 0.6410256410256411, 0.5813953488372093, 0.5681818181818182, 0.49504950495049505, 0.5102040816326531, 0.45045045045045046, 0.4807692307692308, 0.43478260869565216, 0.6172839506172839, 0.43859649122807015, 0.3968253968253968, 0.4032258064516129, 0.3401360544217687, 0.3105590062111801, 0.352112676056338, 0.29069767441860467, 0.352112676056338, 0.24875621890547264, 0.22624434389140272, 0.25773195876288657, 0.25125628140703515, 0.20080321285140562, 0.22935779816513763, 0.2702702702702703, 0.18726591760299627, 0.17921146953405018, 0.11764705882352941, 0.12886597938144329, 0.09615384615384616, 0.09505703422053231, 0.078125, 0.08130081300813008, 0.0748502994011976, 0.09900990099009901, 0.08077544426494346, 0.07451564828614009, 0.0702247191011236, 0.07342143906020558, 0.049800796812749, 0.05005005005005005, 0.03387533875338753, 0.036337209302325583, 0.030864197530864196, 0.03280839895013123, 0.024838549428713365, 0.028604118993135013, 0.021431633090441493, 0.014334862385321102, 0.017674089784376106, 0.014425851125216388, 0.015552099533437015, 0.009288500835965075, 0.006615506747816882, 0.0074827895839568994, 0.007178750897343862, 0.006915629322268326, 0.0050776886361328325, 0.004899559039686428, 0.003439499208915182, 0.003983111606787222, 0.002762278327164245, 0.0018859384429692214, 0.001965872454195172, 0.001438517751309051, 0.001374608236652554, 0.0011133625776570397, 0.0008370722560771445, 0.0006047484851050448, 0.0005154851746463772, 0.0003779889476031721, 0.00041072147334006914, 0.00035631823494199137, 0.00022323122736993433, 0.00017012010479398456]
ber_gldpc_32_28_cc_Ham16_11_v2 = [0.09375, 0.0798611111111111, 0.0673076923076923, 0.05787037037037037, 0.07087053571428571, 0.06308962264150944, 0.08607456140350878, 0.07465277777777778, 0.06414473684210527, 0.05970982142857143, 0.08275462962962964, 0.064453125, 0.06147540983606557, 0.05555555555555555, 0.06303879310344827, 0.06009615384615385, 0.06979166666666667, 0.05320945945945946, 0.05501760563380282, 0.06101190476190476, 0.054464285714285715, 0.06720890410958905, 0.06103515625, 0.04285714285714286, 0.057412790697674417, 0.06949013157894737, 0.054044117647058826, 0.057291666666666664, 0.051962209302325583, 0.052201704545454544, 0.04733910891089109, 0.045599489795918366, 0.05152027027027027, 0.04326923076923077, 0.04782608695652174, 0.05941358024691358, 0.03892543859649123, 0.03521825396825397, 0.03881048387096774, 0.03741496598639456, 0.0343555900621118, 0.034991197183098594, 0.02525436046511628, 0.029049295774647887, 0.026274875621890546, 0.02531108597285068, 0.02851159793814433, 0.025282663316582913, 0.021460843373493976, 0.02236238532110092, 0.03125, 0.01977996254681648, 0.01881720430107527, 0.011764705882352941, 0.01401417525773196, 0.010516826923076924, 0.009565114068441065, 0.008935546875, 0.008079268292682927, 0.008514221556886227, 0.010581683168316832, 0.008531906300484653, 0.008988450074515649, 0.00741748595505618, 0.007755139500734215, 0.005011205179282868, 0.005286536536536536, 0.003662771002710027, 0.004269622093023256, 0.0033179012345679014, 0.0032193241469816274, 0.002856433184302037, 0.003092820366132723, 0.00237087441063009, 0.0014782826834862386, 0.001690084835630965, 0.00148766589728794, 0.0013705287713841369, 0.000981097900798811, 0.0007731873511510982, 0.0006547440885962286, 0.0006864680545585068, 0.0007002074688796681, 0.0005363308621915304, 0.0004991425771680548, 0.0003052555547912224, 0.0004978889508484028, 0.0002555107452626927, 0.00018977255582377793, 0.00021993198081308484, 0.0001483471431037459, 0.00014003821410897894, 0.00011759892226502483, 8.056820464742516e-05, 6.463249434560166e-05, 4.96154480597138e-05, 3.9216353313829106e-05, 4.184225009651955e-05, 3.29594367321342e-05, 2.2323122736993433e-05, 1.7543635806879657e-05]
plt.plot(esno_gldpc_32_28_cc_Ham16_11_v2, fer_gldpc_32_28_cc_Ham16_11_v2, label="GLDPC_32_28_from_Ham16_11_v2", alpha=1, linewidth=2)
# plt.plot(esno_gldpc_32_28_cc_Ham16_11_v2, gaussian_filter1d(fer_gldpc_32_28_cc_Ham16_11_v2, sigma=1).tolist(), label="GLDPC_32_28_from_Ham16_11", alpha=1, linewidth=2)

#----H_ldpc_1---------codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
esno_gldpc_32_28_cc_Ham16_11_v3 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
fer_gldpc_32_28_cc_Ham16_11_v3 = [0.9259259259259259, 0.9615384615384616, 0.8771929824561403, 0.9259259259259259, 0.9090909090909091, 0.8928571428571429, 0.8620689655172413, 0.9433962264150944, 0.819672131147541, 0.9259259259259259, 0.8064516129032258, 0.8064516129032258, 0.847457627118644, 0.8064516129032258, 0.8620689655172413, 0.7692307692307693, 0.819672131147541, 0.7575757575757576, 0.7692307692307693, 0.78125, 0.6944444444444444, 0.7575757575757576, 0.684931506849315, 0.6410256410256411, 0.5813953488372093, 0.6578947368421053, 0.7142857142857143, 0.746268656716418, 0.6578947368421053, 0.6097560975609756, 0.6172839506172839, 0.6097560975609756, 0.6172839506172839, 0.4166666666666667, 0.4807692307692308, 0.46296296296296297, 0.5154639175257731, 0.44642857142857145, 0.5154639175257731, 0.3937007874015748, 0.34965034965034963, 0.3597122302158273, 0.3105590062111801, 0.373134328358209, 0.3246753246753247, 0.3816793893129771, 0.25510204081632654, 0.30864197530864196, 0.2242152466367713, 0.25510204081632654, 0.1893939393939394, 0.18518518518518517, 0.21367521367521367, 0.13966480446927373, 0.17482517482517482, 0.16778523489932887, 0.15151515151515152, 0.1179245283018868, 0.1282051282051282, 0.09727626459143969, 0.10121457489878542, 0.07352941176470588, 0.07874015748031496, 0.10893246187363835, 0.06150061500615006, 0.06925207756232687, 0.055248618784530384, 0.04032258064516129, 0.05055611729019211, 0.03816793893129771, 0.031545741324921134, 0.040749796251018745, 0.0248015873015873, 0.023073373327180433, 0.016490765171503958, 0.018754688672168042, 0.01661681621801263, 0.013743815283122594, 0.009216589861751152, 0.009982032341784788, 0.007460459564309161, 0.009394964299135663, 0.006510416666666667, 0.0058370301190754145, 0.005292125317527519, 0.0037252272388615705, 0.0036342491641226924, 0.0033090668431502318, 0.0021987686895338612, 0.002007387184840212, 0.0024153422539973913, 0.002040233402701269, 0.0011928334565927906, 0.0008859279209043553, 0.0008790899660671273, 0.0007760962359332557, 0.0007139695277805543, 0.00043736878936319107, 0.0003560112499554986, 0.0003191401088906052, 0.0002081806675104923]
ber_gldpc_32_28_cc_Ham16_11_v3 = [0.17881944444444445, 0.21213942307692307, 0.19298245614035087, 0.18344907407407407, 0.1806818181818182, 0.19419642857142858, 0.15140086206896552, 0.20813679245283018, 0.15881147540983606, 0.19039351851851852, 0.14969758064516128, 0.15826612903225806, 0.149364406779661, 0.1350806451612903, 0.15732758620689655, 0.13701923076923078, 0.1413934426229508, 0.11931818181818182, 0.13990384615384616, 0.1328125, 0.1072048611111111, 0.12452651515151515, 0.11215753424657535, 0.10857371794871795, 0.08103197674418605, 0.09251644736842106, 0.11160714285714286, 0.10587686567164178, 0.10731907894736842, 0.08879573170731707, 0.08719135802469136, 0.0819359756097561, 0.09452160493827161, 0.05963541666666667, 0.07421875, 0.06510416666666667, 0.08118556701030928, 0.05915178571428571, 0.061855670103092786, 0.054133858267716536, 0.0465472027972028, 0.05755395683453238, 0.03940217391304348, 0.053404850746268655, 0.040381493506493504, 0.05271946564885496, 0.02646683673469388, 0.03896604938271605, 0.03181053811659193, 0.04193239795918367, 0.021425189393939392, 0.020717592592592593, 0.029513888888888888, 0.018592877094972066, 0.019777097902097904, 0.020448825503355705, 0.019223484848484847, 0.013266509433962265, 0.016506410256410257, 0.01270671206225681, 0.014106781376518218, 0.00942095588235294, 0.009694881889763779, 0.01184640522875817, 0.00730319803198032, 0.009825138504155125, 0.007009668508287293, 0.004259072580645161, 0.005592770475227503, 0.003816793893129771, 0.003529179810725552, 0.004737163814180929, 0.0033172123015873015, 0.0024659667743424088, 0.0018345976253298154, 0.002227119279819955, 0.0016824526420737786, 0.0012369433754810335, 0.0009158986175115207, 0.0010980235575963266, 0.0006900925096985975, 0.0009394964299135663, 0.0005900065104166666, 0.0008098879290217137, 0.0005292125317527519, 0.00045634033676054237, 0.0005360517517080971, 0.00030608868299139644, 0.00025148416886543537, 0.00023837722819977517, 0.0002339862808559973, 0.00021039906965356837, 0.00013195720113557744, 0.00011904656437152274, 9.065615275067251e-05, 7.469926270857587e-05, 7.273564564264398e-05, 5.1937543736878935e-05, 3.471109687066111e-05, 3.909466333909913e-05, 2.2769760508960096e-05]
plt.plot(esno_gldpc_32_28_cc_Ham16_11_v3, fer_gldpc_32_28_cc_Ham16_11_v3, label="GLDPC_32_28_from_Ham16_11_v3", alpha=1, linewidth=2)
# plt.plot(esno_gldpc_32_28_cc_Ham16_11_v3, gaussian_filter1d(fer_gldpc_32_28_cc_Ham16_11_v3, sigma=1).tolist(), label="GLDPC_32_28_from_Ham16_11v3", alpha=1, linewidth=2)

#----H_ldpc_4--------- codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
esno_gldpc_32_28_cc_Ham16_11_v4 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7 ] #9.8, 9.9, 10.0
fer_gldpc_32_28_cc_Ham16_11_v4 = [0.8064516129032258, 0.819672131147541, 0.8333333333333334, 0.746268656716418, 0.7936507936507936, 0.6578947368421053, 0.6756756756756757, 0.7352941176470589, 0.6944444444444444, 0.7575757575757576, 0.6756756756756757, 0.5952380952380952, 0.6024096385542169, 0.5882352941176471, 0.5882352941176471, 0.6410256410256411, 0.4807692307692308, 0.6493506493506493, 0.49019607843137253, 0.49504950495049505, 0.4716981132075472, 0.4807692307692308, 0.44642857142857145, 0.4032258064516129, 0.46296296296296297, 0.44642857142857145, 0.32051282051282054, 0.32051282051282054, 0.32051282051282054, 0.30120481927710846, 0.2347417840375587, 0.37593984962406013, 0.29411764705882354, 0.33783783783783783, 0.2127659574468085, 0.2564102564102564, 0.17301038062283736, 0.2127659574468085, 0.17793594306049823, 0.1278772378516624, 0.1457725947521866, 0.14204545454545456, 0.12224938875305623, 0.1347708894878706, 0.12468827930174564, 0.12254901960784313, 0.09276437847866419, 0.06993006993006994, 0.09746588693957114, 0.07062146892655367, 0.062034739454094295, 0.05213764337851929, 0.04042037186742118, 0.04562043795620438, 0.04205214465937763, 0.03977724741447892, 0.03399048266485384, 0.038314176245210725, 0.025265285497726123, 0.025510204081632654, 0.019692792437967704, 0.02321262766945218, 0.02768549280177187, 0.012357884330202669, 0.013513513513513514, 0.010326311441553077, 0.008195377806916898, 0.008544087491455913, 0.008963786303334529, 0.00787897888433659, 0.006833401667350007, 0.005751754285056942, 0.004283022100394038, 0.004020585397233837, 0.005390835579514825, 0.0030669201987364287, 0.0023440063756973418, 0.0018090379536162668, 0.002068166776968895, 0.0016463615409944023, 0.0021778900601097657, 0.001358621814031846, 0.0009824147755182239, 0.001088992464172148, 0.0006583538520283882, 0.0007703804138483583, 0.0007239661763002432, 0.000686162840165228, 0.0003759172380608685, 0.0004762403680385564, 0.0002504119276209364, 0.00030368368307570837, 0.00019069994507841582, 0.00016130697362308366, 0.0001886529050660851, 0.00012654639697098544, 0.0001176224273034588, 6.810430310228722e-05] #0, 0, 0
ber_gldpc_32_28_cc_Ham16_11_v4 = [0.1537298387096774, 0.18032786885245902, 0.15364583333333334, 0.15391791044776118, 0.1463293650793651, 0.12294407894736842, 0.11739864864864864, 0.12362132352941177, 0.10807291666666667, 0.1543560606060606, 0.10346283783783784, 0.09635416666666667, 0.09186746987951808, 0.08713235294117647, 0.09338235294117647, 0.08774038461538461, 0.07572115384615384, 0.09050324675324675, 0.07169117647058823, 0.07332920792079207, 0.0695754716981132, 0.06881009615384616, 0.04966517857142857, 0.051915322580645164, 0.05092592592592592, 0.05412946428571429, 0.035056089743589744, 0.03345352564102564, 0.034054487179487176, 0.038403614457831324, 0.023327464788732395, 0.04111842105263158, 0.026838235294117645, 0.041173986486486486, 0.022606382978723406, 0.023557692307692307, 0.01730103806228374, 0.01848404255319149, 0.017571174377224198, 0.010789641943734015, 0.01403061224489796, 0.012251420454545454, 0.00993276283618582, 0.011202830188679245, 0.012468827930174564, 0.009267769607843137, 0.007305194805194805, 0.0053321678321678325, 0.008771929824561403, 0.005517302259887006, 0.004924007444168734, 0.0044316996871741395, 0.0034862570735650768, 0.003478558394160584, 0.0037321278385197645, 0.003057875894988067, 0.002740482664853841, 0.0029932950191570882, 0.0019896412329459324, 0.0017697704081632653, 0.0015015754233950373, 0.0015668523676880223, 0.0018514673311184938, 0.0009268413247652002, 0.0009459459459459459, 0.0006647562990499793, 0.0005480658908375676, 0.0005660457963089542, 0.0005770437432771603, 0.0005022849038764576, 0.00045271286046193795, 0.0003882434142413436, 0.00027839643652561246, 0.0002512865873271148, 0.00038409703504043124, 0.00019934981291786787, 0.000181660494116544, 0.00012437135931111834, 0.00013184563203176705, 0.00010495554823839315, 0.00014156285390713478, 8.661214064453019e-05, 6.385696040868455e-05, 6.806202901075925e-05, 4.197005806680975e-05, 4.911175138283284e-05, 4.52478860187652e-05, 4.288517751032675e-05, 2.3494827378804282e-05, 3.036032346245797e-05, 1.5963760385834697e-05, 1.8980230192231773e-05, 1.1918746567400989e-05, 1.0081685851442729e-05, 1.1790806566630319e-05, 7.90914981068659e-06, 7.351401706466175e-06, 4.34164932277081e-06] #0, 0, 0
plt.plot(esno_gldpc_32_28_cc_Ham16_11_v3, fer_gldpc_32_28_cc_Ham16_11_v4, label="GLDPC_32_28_from_Ham16_11_v4", alpha=1, linewidth=2)
# plt.plot(esno_gldpc_32_28_cc_Ham16_11_v3, gaussian_filter1d(fer_gldpc_32_28_cc_Ham16_11_v4, sigma=1).tolist(), label="GLDPC_32_28_from_Ham16_11v4", alpha=1, linewidth=2)

plt.yscale("log")  # Логарифмическая шкала по Y
plt.xlabel("EsNo [dB]")
plt.ylabel("FER")
# plt.title("LDPC(32,28)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig('modeling_results/GLDPC(32,28)_LDPC_real_compare.png', dpi=300, bbox_inches='tight')