from src.voxguard.fusion.redflags import RED_FLAG_PHRASES

isolation_entries = [entry for entry in RED_FLAG_PHRASES if entry[1] == "isolation"]
print("Isolation category entries:")
for entry in isolation_entries:
    print(" ", entry)
