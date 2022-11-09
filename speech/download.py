from dataset import SubsetSC

train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

print(f"Train: {len(train_set):,}")
print(f"Val: {len(val_set):,}")
print(f"Test: {len(test_set):,}")
print("Done!")