"""데이터 검증 스크립트: python -m src.check_data

모든 ID가 고유한지, 모든 데이터가 스키마를 통과하는지, 고객 3명의 담보 구성이 서로 다른지 출력한다.
"""
from src import tools


def main() -> None:
    summary = tools.validate_data()  # 스키마 검증은 로드할 때 함께 한다
    print("스키마 검증: 통과 | ID 고유: 통과 |", " / ".join(f"{k} {v}개" for k, v in summary.items()))
    print("\n[고객별 담보 구성]")
    for c in tools.load_customers():
        joined = ", ".join(
            f"{k}({'한도 없음' if v.limit is None else f'{v.limit:,}원'})" for k, v in c.coverages.items() if v.joined
        )
        not_joined = [k for k, v in c.coverages.items() if not v.joined]
        print(f"- {c.id} {c.name}: {joined}" + (f" | 미가입: {', '.join(not_joined)}" if not_joined else "")
              + (f" | 특약: {', '.join(c.riders)}" if c.riders else ""))
    print("\n[기준표: 고객 과실 범위]")
    for e in tools.load_fault_table():
        mods = ", ".join(f"{m.id} {m.bound}{m.adjustment:+d}" for m in e.modifiers)
        print(f"- {e.id} {e.accident_type}: {e.customer_fault.min}~{e.customer_fault.max}% | {mods}")


if __name__ == "__main__":
    main()
