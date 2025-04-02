from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt


esno_ldpc_32_28 = [2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
fer_ldpc_32_28 = [0.9433962264150944, 0.9615384615384616, 0.9259259259259259, 0.9090909090909091, 0.8928571428571429, 0.8771929824561403, 0.8064516129032258, 0.7246376811594203, 0.746268656716418, 0.6410256410256411, 0.6329113924050633, 0.625, 0.5434782608695652, 0.5263157894736842, 0.43478260869565216, 0.4, 0.32051282051282054, 0.33783783783783783, 0.2840909090909091, 0.26737967914438504, 0.1736111111111111, 0.20833333333333334, 0.12594458438287154, 0.15873015873015872, 0.09276437847866419, 0.06657789613848203, 0.046948356807511735, 0.051387461459403906, 0.03924646781789639, 0.033112582781456956, 0.02485089463220676]
ber_ldpc_32_28 = [0.044221698113207544, 0.046875, 0.05497685185185185, 0.04772727272727273, 0.04743303571428571, 0.047149122807017545, 0.038306451612903226, 0.036684782608695655, 0.04757462686567164, 0.038461538461538464, 0.03560126582278481, 0.037109375, 0.030570652173913044, 0.03125, 0.028532608695652172, 0.0235, 0.021033653846153848, 0.02153716216216216, 0.018465909090909092, 0.01771390374331551, 0.010091145833333334, 0.013411458333333333, 0.007399244332493703, 0.010218253968253968, 0.006319573283858998, 0.004327563249001331, 0.003051643192488263, 0.0031153648509763617, 0.0024529042386185244, 0.0020281456953642383, 0.0015531809145129226]
plt.plot(esno_ldpc_32_28, fer_ldpc_32_28, label="LDPC(32,28)", alpha=1, linewidth=2)
# plt.plot(esno_ldpc_32_28, gaussian_filter1d(fer_ldpc_32_28, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

esno_gldpc = [2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
fer_gldpc = [0.3546099290780142, 0.35714285714285715, 0.3401360544217687, 0.26455026455026454, 0.21739130434782608, 0.15527950310559005, 0.2358490566037736, 0.11547344110854503, 0.12531328320802004, 0.08620689655172414, 0.1006036217303823, 0.06925207756232687, 0.08503401360544217, 0.04382120946538125, 0.04460303300624442, 0.03692762186115214, 0.03546099290780142, 0.02844141069397042, 0.020366598778004074, 0.019952114924181964, 0.013099292638197537, 0.012254901960784314, 0.008742787200559538, 0.0073551044424830835, 0.005460303592879764, 0.00450815976918222, 0.00406934158053227, 0.002743183189773413, 0.0025040064102564105, 0.0015561295944726277, 0.0015502433882119493]
ber_gldpc = [0.011081560283687944, 0.013169642857142857, 0.012542517006802721, 0.008763227513227513, 0.007201086956521739, 0.005725931677018634, 0.008254716981132075, 0.0040415704387990765, 0.004151002506265664, 0.003556034482758621, 0.003709758551307847, 0.0024238227146814403, 0.002816751700680272, 0.0015337423312883436, 0.0018119982158786797, 0.0016386632200886262, 0.0017065602836879433, 0.0011554323094425483, 0.0008655804480651731, 0.0008853750997605746, 0.0004584752423369138, 0.0004978553921568627, 0.0003059975520195838, 0.00027121947631656367, 0.00021158676422409085, 0.00019159679019024433, 0.00016531700170912346, 0.00011658528556537005, 0.00012050530849358975, 6.613550776508668e-05, 6.007193129321303e-05]
plt.plot(esno_gldpc, fer_gldpc, label="GLDPC", alpha=1, linewidth=2)
# plt.plot(esno_ldpc_32_28, gaussian_filter1d(fer_ldpc_32_28, sigma=1).tolist(), label="LDPC(32,28)", alpha=1, linewidth=2)

plt.yscale("log")  # Логарифмическая шкала по Y
plt.xlabel("EsNo [dB]")
plt.ylabel("FER")
# plt.title("LDPC(32,28)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig('modeling_results/LDPC(32,28)_real.png', dpi=300, bbox_inches='tight')