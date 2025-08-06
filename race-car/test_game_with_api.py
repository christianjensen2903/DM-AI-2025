import sys
import os
import random
import pygame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.game.core import initialize_game_state, game_loop


def test_game_with_api():
    """Test the game with API integration."""
    pygame.init()

    seed = random.randint(1, 1000000)

    print("🚗 Starting race car game with API integration...")
    print("API URL: http://localhost:8000")

    # Initialize the game state with the API URL
    initialize_game_state(
        api_url="http://localhost:8000",
        seed_value=seed,
        sensor_removal=0,  # Keep all sensors
    )

    print("✅ Game state initialized!")
    print("🎮 Starting game loop...")
    print("Press Ctrl+C to stop the game")

    try:
        # Run the game loop
        game_loop(
            verbose=True,  # Show the game window
            log_actions=False,
            log_path="test_actions_log.json",
        )
    except KeyboardInterrupt:
        print("\n🛑 Game stopped by user")
    except Exception as e:
        print(f"❌ Error during game: {e}")
        raise

    print("Seed:", seed)


if __name__ == "__main__":
    test_game_with_api()
