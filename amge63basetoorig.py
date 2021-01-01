def basetoorig(price):
    cert = input("Certified: ")
    if (cert == "y"):
        price = price + 1500
    bodycf = input("AMG Carbon Fiber Package: ")
    if (bodycf == "y"):
        price = price + 1750
    premium = input("Premium Package: ")
    if (premium == "y"):
        price = price + 3600
    night = input("AMG Night Package: ")
    if (night == "y"):
        price = price + 750
    advlight = input("Advanced Lighting Package: ")
    if (advlight == "y"):
        price = price + 1500
    acoustic = input("Acoustic comfort package: ")
    if (acoustic == "y"):
        price = price + 1100
    enginecf = input("AMG Carbon Fiber engine cover: ")
    if (enginecf == "y"):
        price = price + 1500
    mirrorcf = input("AMG Carbon Fiber mirrors: ")
    if (mirrorcf == "y"):
        price = price + 1200
    power = input("115V AC Power Outlet: ")
    if (power == "y"):
        price = price + 115
    frontseats = input("Active multicontour front seats")
    if (frontseats == "y"):
        price = price + 1320
    headliner = input("designo Healiner: ")
    if (headliner == "y"):
        price = price + 1600
    softclose = input("Soft close doors: ")
    if (softclose == "y"):
        price = price + 550
    exhaust = input("AMG Performance Exhaust: ")
    if (exhaust == "y"):
        price = price + 1250
    seats = input("AMG Performance Seats: ")
    if (seats == "y"):
        price = price + 2500
    climatecontrol = input("Three-zone climate control: ")
    if (climatecontrol == "y"):
        price = price + 760
    blacktires = input("Black cross-spoke tires: ")
    if (blacktires == "y"):
        price = price + 1700
    warmth = input("Warmth and comfort package? ")
    if (warmth == "y"):
        price = price + 1050
    driveassist = input("Driver Assistant Package? ")
    if (driveassist == "y"):
        price = price + 2250
    rearseats = input("Heated rear seats? ")
    if (rearseats == "y"):
        price = price + 580
    pano = input("Sun Protection Package: ")
    if (pano == "y"):
        price = price + 380
    air = input("Air Balance Filtration: ")
    if (air == "y"):
        price = price + 550
    extlight = input("Exterior Lighting Package: ")
    if (extlight == "y"):
        price = price + 800
    rearairbags = input("Rear side airbags: ")
    if (rearairbags == "y"):
        price = price + 420
    return price