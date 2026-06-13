import math

# Hunter MP Rotor nozzle database
MP_ROTOR_MODELS = {
    "MP800":  {"radius_min_m": 1.8, "radius_max_m": 3.7,  "precip_rate_mm_hr": 20.0, "optimal_psi": 35},
    "MP1000": {"radius_min_m": 2.5, "radius_max_m": 4.5,  "precip_rate_mm_hr": 10.0, "optimal_psi": 40},
    "MP2000": {"radius_min_m": 4.0, "radius_max_m": 6.4,  "precip_rate_mm_hr": 10.0, "optimal_psi": 40},
    "MP3000": {"radius_min_m": 6.7, "radius_max_m": 9.0,  "precip_rate_mm_hr": 10.0, "optimal_psi": 40},
}

def select_mp_rotor(field_width_m: float) -> dict:
    """Auto-select best MP Rotor model for a given field width (head-to-head spacing)."""
    required_radius = field_width_m / 2.0  # head-to-head coverage rule
    for model, specs in MP_ROTOR_MODELS.items():
        if specs["radius_min_m"] <= required_radius <= specs["radius_max_m"]:
            return {"model": model, **specs}
    raise ValueError(f"No MP Rotor covers radius {required_radius:.1f}m. Use a different head type.")

def compute_irrigation_duration(
    target_depth_mm: float,       # from LSTM ET output
    model_name: str = "MP2000",
    arc_deg: float = 360.0,
    radius_m: float = None,
) -> dict:
    """
    Compute how long to run the MP Rotor to deliver target_depth_mm of water.
    
    Key MP Rotor property: precipitation rate is MATCHED regardless of arc/radius.
    So duration = target_depth / precip_rate (no arc correction needed!)
    """
    specs = MP_ROTOR_MODELS[model_name]
    r = radius_m or specs["radius_max_m"]
    precip_rate = specs["precip_rate_mm_hr"]
    
    duration_hr = target_depth_mm / precip_rate
    duration_min = duration_hr * 60
    
    # Coverage area for one head
    coverage_area_m2 = math.pi * r**2 * (arc_deg / 360.0)
    
    return {
        "model": model_name,
        "target_depth_mm": target_depth_mm,
        "duration_minutes": round(duration_min, 1),
        "radius_m": r,
        "arc_deg": arc_deg,
        "coverage_area_m2": round(coverage_area_m2, 2),
        "precip_rate_mm_hr": precip_rate,
        "optimal_pressure_psi": specs["optimal_psi"],
    }

def compute_zone_coverage(
    field_area_m2: float,
    model_name: str = "MP2000",
    spacing_pattern: str = "square",  # "square" or "triangular"
) -> dict:
    """Calculate number of heads needed and total flow for a field zone."""
    specs = MP_ROTOR_MODELS[model_name]
    r = specs["radius_max_m"]
    
    if spacing_pattern == "square":
        area_per_head = r ** 2   # head-to-head square grid
    else:
        area_per_head = (math.sqrt(3) / 2) * r ** 2  # triangular = denser
    
    num_heads = math.ceil(field_area_m2 / area_per_head)
    return {
        "model": model_name,
        "field_area_m2": field_area_m2,
        "num_heads": num_heads,
        "spacing_pattern": spacing_pattern,
        "area_per_head_m2": round(area_per_head, 2),
    }