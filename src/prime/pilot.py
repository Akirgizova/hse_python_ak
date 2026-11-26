import math
from statistics import NormalDist

z = NormalDist().inv_cdf


# расчёт необходимого размера каждой группы для заданного MDE
def n_per_group(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                comparisons: int = 1) -> int:
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


# расчёт минимального детектируемого эффекта для заданного размера группы
def mde_for_n(sigma: float, n: int, alpha: float = 0.05, power: float = 0.8,
              comparisons: int = 1) -> float:
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return (za + zb) * sigma * math.sqrt(2 / n)


# разность средних test - control, доверительный интервал и p-значение по Уэлчу
def diff_ci(t, c, alpha: float = 0.05) -> dict:
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5
    q = NormalDist().inv_cdf(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


# проверка соотношения размеров групп (SRM) для 2 или 3 групп без scipy
def srm_pvalue(sizes) -> float:
    n = sum(sizes)
    exp = n / len(sizes)
    chi2 = sum((s - exp) ** 2 / exp for s in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("для 4 и более групп — scipy.stats.chisquare")


# поправка CUPED: возвращает скорректированную метрику и коэффициент theta
def cuped(y, x):
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta