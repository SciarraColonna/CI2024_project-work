import numpy as np

# safe log function necessary for handling the logarithm of negative values
EPSILON = 1e-10
def safe_log(val):
    return np.log(np.maximum(val, EPSILON))



def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.square(
            np.multiply(
                safe_log(np.subtract(
                    -4.216101004052772,
                    x[0]
                )),
                safe_log(np.add(
                    -4.643831930033768,
                    x[0]
                ))
            )
        ),
        np.add(
            np.add(
                np.add(
                    np.add(
                        x[1],
                        x[0]
                    ),
                    np.add(
                        x[0],
                        x[1]
                    )
                ),
                np.add(
                    np.multiply(
                        x[2],
                        1.2463374665659082
                    ),
                    np.multiply(
                        x[0],
                        1.2116846064144768
                    )
                )
            ),
            np.multiply(
                x[0],
                1.2463374665659082
            )
        )
    )

def f3(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.subtract(
            np.add(
                np.subtract(
                    np.square(x[0]),
                    np.add(
                        x[2],
                        -0.7156700563515184
                    )
                ),
                np.subtract(
                    np.square(x[0]),
                    np.add(
                        x[2],
                        x[2]
                    )
                ),
            ),
            np.multiply(
                np.square(x[1]),
                x[1]
            ),
        ),
        3.3239057976169306
    )

def f4(x: np.ndarray) -> np.ndarray:
    return np.divide(
        np.divide(
            np.add(
                np.divide(
                    -3.6998018083898976,
                    x[1]
                ),
                np.divide(
                    x[1],
                    2.1614344017503093
                )
            ),
            np.divide(
                np.divide(
                    np.exp(-1.873184645375078),
                    np.sin(1.8111621637415212)
                ),
                np.sin(
                    np.add(
                        x[1],
                        x[1]
                    )
                )
            )
        ),
        -3.9544220632203286
    )

def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.sqrt(np.abs(
            np.subtract(
                x[1],
                x[1]
            )
        )),
        np.subtract(
            np.multiply(
                np.subtract(
                    np.multiply(
                        x[1],
                        x[0]
                    ),
                    np.sin(x[0])
                ),
                np.subtract(
                    np.subtract(
                        3.86355613819784,
                        x[1]
                    ),
                    np.sqrt(np.abs(
                        x[0]
                    ))
                )
            ),
            np.subtract(
                np.multiply(
                    np.multiply(
                        -2.3978566620240427,
                        x[1]
                    ),
                    np.sin(x[0])
                ),
                np.sqrt(np.abs(
                    np.multiply(
                        x[1],
                        x[0]
                    )
                ))
            )
        )
    )

def f6(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.add(
            np.divide(
                np.subtract(x[1], x[0]),
                -2.9049004649149843
            ),
            x[1]
        ),
        np.subtract(x[1], x[0])
    )

def f7(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.sqrt(np.abs(
            np.multiply(
                x[0],
                np.add(
                    np.multiply(
                        x[1],
                        2.277723115439093
                    ),
                    np.multiply(
                        x[1],
                        2.277723115439093
                    )
                )
            )
        )),
        np.add(
            np.add(
                np.multiply(
                    np.sin(x[0]),
                    x[1]
                ),
                np.sin(
                    np.multiply(
                        x[0],
                        x[0]
                    )
                )
            ),
            np.add(
                np.exp(
                    np.multiply(
                        x[0],
                        x[1]
                    )
                ),
                np.sqrt(np.abs(
                    -3.248470480777985
                ))
            )
        )
    )

def f8(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.multiply(
            np.add(
                np.add(
                    np.multiply(
                        x[5],
                        4.734189255288326
                    ),
                    np.multiply(
                        x[5],
                        4.734189255288326
                    )
                ),
                np.add(
                    np.add(
                        x[5],
                        x[5]
                    ),
                    x[5]
                )
            ),
            np.subtract(
                np.subtract(
                    np.tan(-4.727738110312435),
                    np.multiply(
                        x[5],
                        4.734189255288326
                    )
                ),
                np.add(
                    np.multiply(
                        x[5],
                        4.734189255288326
                    ),
                    np.multiply(
                        x[5],
                        4.734189255288326
                    )
                )
            )
        ),
        np.add(
            safe_log(
                np.tan(-4.727738110312435)
            ),
            np.exp(
                np.add(
                    x[5],
                    x[5]
                )
            )
        )
    )