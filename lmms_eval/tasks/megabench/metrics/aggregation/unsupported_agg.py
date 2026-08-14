from numbers import Number


class UnsupportedAggregation:
    @staticmethod
    def aggregate(scores: dict[str, Number], weights: dict[str, Number]) -> Number:
        return -1
