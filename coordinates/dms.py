from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Coordinate:
    deg: int
    min: int
    sec: int
    negative: bool = False

    def __str__(self):
        sign = "-" if self.negative else ""
        return f"{sign}{self.deg}° {self.min}' {self.sec}\""

    def to_decimal(self):
        decimal = self.deg + self.min / 60 + self.sec / 3600
        return -decimal if self.negative else decimal

    def copy_with_negation(self):
        return Coordinate(
            deg=self.deg,
            min=self.min,
            sec=self.sec,
            negative=not self.negative
        )


class Coords(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def convert_to(self, target_type: type) -> 'Coords':
        pass


@dataclass
class CoordsNE(Coords):
    N: Coordinate
    E: Coordinate

    def __str__(self):
        return f"N: {self.N}, E: {self.E}"

    def convert_to(self, target_type: type) -> Coords:
        if target_type is CoordsNW:
            return CoordsNW(N=self.N, W=self.E.copy_with_negation())
        return self


@dataclass
class CoordsNW(Coords):
    N: Coordinate
    W: Coordinate

    def __str__(self):
        return f"N: {self.N}, W: {self.W}"

    def convert_to(self, target_type: type) -> Coords:
        if target_type is CoordsNE:
            return CoordsNE(N=self.N, E=self.W.copy_with_negation())
        return self