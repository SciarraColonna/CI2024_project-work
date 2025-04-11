import numpy as np



def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.multiply(
            np.multiply(
                np.exp(
                    np.square(3.6395654523436693)
                ),
                np.add(
                    x[2],
                    x[1]
                )
            ),
            np.multiply(
                np.cos(
                    np.sqrt(np.abs(x[0]))
                ),
                2.6782220403840427
            )
        ),
        np.multiply(
            np.multiply(
                5.054900013234663,
                np.exp(
                    np.exp(
                        2.5100936565552363
                    )
                )
            ),
            x[0]
        )
    )


def f3(x: np.ndarray) -> np.ndarray:
    return np.subtract(
        np.add(
            np.add(
                np.sqrt(np.abs(
                    np.square(-1.3764633657775143)
                )),
                np.square(x[0])
            ),
            np.subtract(
                np.subtract(
                    np.square(x[0]),
                    np.multiply(
                        3.5018153531861316,
                        x[2]
                    )
                ),
                np.subtract(
                    np.cos(-1.2027135079839026),
                    2.9832682152237946
                )
            )
        ),
        np.multiply(
            x[1],
            np.square(x[1])
        )
    )


def f4(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.multiply(
            np.add(
                4.946167613668001,
                np.sqrt(np.abs(-4.20850453732937))
            ),
            np.log(
                np.exp(
                    np.cos(x[1])
                )
            )
        ),
        3.279850285907879
    )


def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.multiply(
            np.multiply(
                0.09079503875486852,
                np.divide(
                    np.exp(x[0]),
                    -4.380200442350067
                )
            ),
            np.sqrt(np.abs(
                np.square(
                    np.exp(x[1])
                )
            ))
        ),
        np.square(
            -7.062896607124003e-06
        )
    )


def f6(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.subtract(
            np.multiply(
                np.subtract(
                    np.square(0.8879430519811904),
                    -1.2085705848982418
                ),
                x[1]
            ),
            np.multiply(
                np.sqrt(np.abs(-0.6691830904317395)),
                x[0]
            )
        ),
        0.8484846659054726
    )


def f7(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.exp(
            np.multiply(
                x[0],
                x[1]
            )
        ),
        np.multiply(
            -1.4637465573496258,
            np.log(
                np.square(
                    np.subtract(
                        x[0],
                        x[1]
                    )
                )
            )
        )
    )


def f8(x: np.ndarray) -> np.ndarray:
    return np.subtract(
        np.divide(
            np.square(
                np.exp(x[5])
            ),
            np.exp(
                np.cos(x[5])
            )
        ),
        np.add(
            np.multiply(
                np.exp(
                    np.subtract(
                        1.9127822778977563,
                        x[5]
                    )
                ),
                np.square(
                    -3.847658434677139
                )
            ),
            np.square(
                np.divide(
                    np.square(x[4]),
                    np.sin(
                        -0.5462398013115592
                    )
                )
            )
        )
    )