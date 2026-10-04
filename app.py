from model import recommend_for_user, load_data

def main():
    print("=" * 72)
    print("SERVICE RECOMMENDATION SYSTEM - TENSORFLOW")
    print("=" * 72)
    print("Existing customer examples: U001, U002, U003, ... U008")
    print("A new ID such as U100 demonstrates cold-start handling.\n")

    services, _ = load_data()
    categories = sorted(services["category"].unique())
    print("Available categories:")
    for category in categories:
        print(f"  - {category}")

    while True:
        user_id = input("\nCustomer ID (or 'exit'): ").strip()
        if user_id.lower() == "exit":
            print("Goodbye.")
            break
        if not user_id:
            continue

        category = input("Optional category [Enter = all]: ").strip() or None
        top_n_text = input("Number of recommendations [5]: ").strip()
        try:
            top_n = int(top_n_text) if top_n_text else 5
            top_n = max(1, min(top_n, 20))
        except ValueError:
            top_n = 5

        results, cold_start = recommend_for_user(
            user_id=user_id, top_n=top_n, category=category
        )

        if results.empty:
            print("\nNo recommendations matched the selected category.")
            continue

        if cold_start:
            print("\nNew customer: using popularity/ratings as fallback.")
            for i, row in enumerate(results.itertuples(), 1):
                print(
                    f"{i}. {row.service_name} | {row.category} | "
                    f"${row.price:.2f} | average={row.avg_rating:.2f}"
                )
        else:
            print("\nPersonalized recommendations:")
            for i, row in enumerate(results.itertuples(), 1):
                print(
                    f"{i}. {row.service_name} | {row.category} | "
                    f"${row.price:.2f} | prediction={row.predicted_score:.2f}/5"
                )

if __name__ == "__main__":
    main()
