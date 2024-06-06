class MathUtils:
    @staticmethod
    def gcd(a, b):
        """求两个数的最大公约数"""
        while b != 0:
            a, b = b, a % b
        return a

    @classmethod
    def lcm(cls, a, b):
        """求两个数的最小公倍数"""
        return a * b // cls.gcd(a, b)
