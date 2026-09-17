def get_congestion_level(occupancy_percent):
    if occupancy_percent < 40:
        return "Low"
    elif occupancy_percent <= 75:
        return "Moderate"
    else:
        return "High"


def get_recommendation(occupancy_percent):
    if occupancy_percent >= 90:
        return "Parking full — try another area."
    elif occupancy_percent >= 75:
        return "Parking nearly full — hurry, limited slots available."
    else:
        return "Slots available — proceed to park."


def summarize(total_slots, occupied_slots):
    available_slots = total_slots - occupied_slots
    occupancy_percent = (occupied_slots / total_slots) * 100 if total_slots > 0 else 0
    return {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "available_slots": available_slots,
        "occupancy_percent": round(occupancy_percent, 1),
        "congestion_level": get_congestion_level(occupancy_percent),
        "recommendation": get_recommendation(occupancy_percent),
    }
