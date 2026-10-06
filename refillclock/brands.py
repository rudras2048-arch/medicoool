"""Indian retail pharma brand mapping and margin assistance for chronic salts."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BrandOption:
    brand_name: str
    manufacturer: str
    mrp_per_strip: float
    chemist_margin_pct: float
    is_generic: bool

# Brand equivalence dictionary for top chronic salts in India
BRAND_CATALOG: dict[str, list[BrandOption]] = {
    "Metformin 500 mg": [
        BrandOption("Glycomet 500", "USV", 48.50, 18.0, False),
        BrandOption("Cetapin 500", "Sanofi", 52.00, 19.5, False),
        BrandOption("Bigomet 500", "Aristo", 42.00, 22.0, False),
        BrandOption("Jan Aushadhi Metformin", "BPPI (Generic)", 14.50, 40.0, True),
    ],
    "Telmisartan 40 mg": [
        BrandOption("Telma 40", "Glenmark", 125.00, 18.5, False),
        BrandOption("Telmikind 40", "Mankind", 85.00, 24.0, False),
        BrandOption("Arbitel 40", "Micro Labs", 92.00, 22.0, False),
        BrandOption("Jan Aushadhi Telmisartan", "BPPI (Generic)", 28.00, 42.0, True),
    ],
    "Atorvastatin 10 mg": [
        BrandOption("Atorva 10", "Zydus Cadila", 145.00, 19.0, False),
        BrandOption("Storvas 10", "Sun Pharma", 160.00, 18.0, False),
        BrandOption("Lipikind 10", "Mankind", 95.00, 25.0, False),
        BrandOption("Jan Aushadhi Atorvastatin", "BPPI (Generic)", 32.00, 45.0, True),
    ],
    "Amlodipine 5 mg": [
        BrandOption("Amlong 5", "Micro Labs", 54.00, 20.0, False),
        BrandOption("Stamlo 5", "Dr. Reddy's", 68.00, 18.0, False),
        BrandOption("Amlovas 5", "Macleods", 48.00, 22.5, False),
        BrandOption("Jan Aushadhi Amlodipine", "BPPI (Generic)", 12.00, 38.0, True),
    ],
    "Levothyroxine 50 mcg": [
        BrandOption("Thyronorm 50", "Abbott", 185.00, 17.5, False),
        BrandOption("Eltroxin 50", "GSK", 175.00, 18.0, False),
        BrandOption("Thyrox 50", "Macleods", 140.00, 22.0, False),
    ],
    "Glimepiride 1 mg": [
        BrandOption("Amaryl 1", "Sanofi", 112.00, 18.0, False),
        BrandOption("Glimestar 1", "Mankind", 55.00, 25.0, False),
        BrandOption("Gepiride 1", "Micro Labs", 62.00, 22.0, False),
        BrandOption("Jan Aushadhi Glimepiride", "BPPI (Generic)", 16.00, 40.0, True),
    ],
}

def get_substitutes(salt_name: str) -> list[BrandOption]:
    return BRAND_CATALOG.get(salt_name, [
        BrandOption(f"{salt_name} (Standard Brand)", "Pharma Co", 65.00, 20.0, False),
        BrandOption(f"{salt_name} (Jan Aushadhi)", "Generic", 18.00, 40.0, True),
    ])
