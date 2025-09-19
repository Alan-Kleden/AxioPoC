from appetition_aversion.regulator import appetition_aversion_score

def run_case(fc, fi):
    score = appetition_aversion_score(fc, fi)
    print(f"Fc={fc:.2f}  Fi={fi:.2f}  =>  net={score:.2f}  ({'approach' if score>0 else 'avoid' if score<0 else 'ambivalent'})")

if __name__ == "__main__":
    print("== Appetition–Aversion demo ==")
    run_case(2.0, 1.0)
    run_case(1.0, 3.0)
    run_case(1.5, 1.5)
