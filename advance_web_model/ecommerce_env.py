# ecommerce_env.py (Updated with fixes)
from __future__ import annotations
import time
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import random


class EcommerceEnv(gym.Env):
    """
    Environment for e-commerce with login and payment.
    Agent must: login -> browse -> add to cart -> checkout -> pay
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    # Hardcoded credentials
    VALID_USERNAME = "admin"
    VALID_PASSWORD = "password123"

    def __init__(
        self,
        app_url: str = "http://localhost:8080",
        headless: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        target_items: int = 2
    ):
        super().__init__()
        self.app_url = app_url
        self.headless = headless
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target_items = target_items

        # Actions: 
        # 0=type_username, 1=type_password, 2=click_login,
        # 3=add_laptop, 4=add_headphones, 5=add_mouse, 6=add_keyboard,
        # 7=remove_item, 8=proceed_to_checkout,
        # 9=type_card_name, 10=type_card_number, 11=type_expiry, 12=type_cvv,
        # 13=type_address, 14=type_city, 15=type_zip, 16=complete_payment
        self.action_space = spaces.Discrete(17)

        # Observation: [is_logged_in, items_in_cart, on_store_page, on_checkout_page,
        #               username_filled, card_fields_progress, address_fields_progress, 
        #               steps_ratio, payment_completed, page_encoding]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        self.driver = None
        self.steps = 0
        self.is_logged_in = False
        self.items_in_cart = 0
        self.username_filled = False
        self.password_filled = False
        self.card_name_filled = False
        self.card_number_filled = False
        self.expiry_filled = False
        self.cvv_filled = False
        self.address_filled = False
        self.city_filled = False
        self.zip_filled = False
        self.payment_completed = False
        self.current_page = "login"  # login, store, checkout
        
        # Track if driver needs recreation
        self.driver_alive = False

    def _create_driver(self):
        """Create a new Chrome driver instance."""
        try:
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-tools")
            chrome_options.add_argument("--remote-debugging-port=0")  # Avoid port conflicts
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(10)
            self.driver.implicitly_wait(0.5)
            self.driver_alive = True
            return True
        except Exception as e:
            print(f"Failed to create driver: {e}")
            self.driver_alive = False
            return False

    def _check_driver_alive(self):
        """Check if driver is still alive, recreate if not."""
        try:
            if not self.driver or not self.driver_alive:
                return self._create_driver()
            # Try a simple command to check if driver is responsive
            self.driver.current_url
            return True
        except (WebDriverException, Exception) as e:
            print(f"Driver not responsive: {e}. Recreating...")
            self.driver_alive = False
            return self._create_driver()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Ensure driver is alive
        if not self._check_driver_alive():
            # If we can't create driver, return a safe observation
            print("ERROR: Could not create driver in reset()")
            return self._get_obs(), self._get_info()
        
        try:
            self.driver.get(self.app_url)
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed to load page: {e}")
            self.driver_alive = False
            return self._get_obs(), self._get_info()

        # Reset state
        self.steps = 0
        self.is_logged_in = False
        self.items_in_cart = 0
        self.username_filled = False
        self.password_filled = False
        self.card_name_filled = False
        self.card_number_filled = False
        self.expiry_filled = False
        self.cvv_filled = False
        self.address_filled = False
        self.city_filled = False
        self.zip_filled = False
        self.payment_completed = False
        self.current_page = "login"

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # Check if driver is alive before doing anything
        if not self._check_driver_alive():
            print("Driver died during step, ending episode")
            return self._get_obs(), -10.0, True, False, self._get_info()

        try:
            # LOGIN ACTIONS
            if action == 0:  # Type username
                if not self.is_logged_in and not self.username_filled:
                    try:
                        username_field = self.driver.find_element(By.ID, "username")
                        username_field.clear()
                        username_field.send_keys(self.VALID_USERNAME)
                        self.username_filled = True
                        reward += 2.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"username action failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 1:  # Type password
                if not self.is_logged_in and not self.password_filled:
                    try:
                        password_field = self.driver.find_element(By.ID, "password")
                        password_field.clear()
                        password_field.send_keys(self.VALID_PASSWORD)
                        self.password_filled = True
                        reward += 2.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"password action failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 2:  # Click login
                if not self.is_logged_in and self.username_filled and self.password_filled:
                    try:
                        login_btn = self.driver.find_element(By.ID, "loginBtn")
                        login_btn.click()
                        time.sleep(0.5)
                        self.driver.find_element(By.ID, "storeScreen")
                        self.is_logged_in = True
                        self.current_page = "store"
                        reward += 10.0
                    except Exception as e:
                        print(f"login click failed: {e}")
                        reward -= 5.0
                else:
                    reward -= 1.0

            # SHOPPING ACTIONS
            elif action in (3, 4, 5, 6):  # Add specific product
                if self.is_logged_in and self.current_page == "store":
                    before_count = self._count_cart_items()
                    index = action - 3
                    clicked = self._click_add_to_cart(index)
                    time.sleep(0.05)
                    after_count = self._count_cart_items()

                    if clicked and after_count > before_count:
                        marginal = max(0.5, 4.0 - 0.8 * after_count)
                        reward += marginal
                        if after_count > self.target_items:
                            reward -= 1.0
                    else:
                        reward -= 1.0
                else:
                    reward -= 0.5

            elif action == 7:  # Remove item
                if self.is_logged_in and self.current_page == "store":
                    try:
                        remove_buttons = self.driver.find_elements(By.CLASS_NAME, "btn-remove")
                        if remove_buttons:
                            remove_buttons[0].click()
                            time.sleep(0.05)
                            reward += 0.5
                        else:
                            reward -= 0.5
                    except Exception as e:
                        print(f"remove action failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 8:  # Proceed to checkout
                if self.is_logged_in and self.current_page == "store":
                    cart_count = self._count_cart_items()
                    if cart_count == 0:
                        reward -= 1.0
                    else:
                        try:
                            checkout_btn = self.driver.find_element(By.ID, "checkoutBtn")
                            checkout_btn.click()
                            time.sleep(0.5)
                            self.driver.find_element(By.ID, "checkoutScreen")
                            self.current_page = "checkout"
                            reward += 8.0 + (5.0 if cart_count >= self.target_items else 0.0)
                        except Exception as e:
                            print(f"checkout click failed: {e}")
                            reward -= 3.0
                else:
                    reward -= 1.0

            # PAYMENT ACTIONS - Type full field values at once
            elif action == 9:  # Type card name (full name)
                if self.current_page == "checkout" and not self.card_name_filled:
                    try:
                        full_name = random.choice(["John Doe", "Jane Smith", "Alice Johnson", "Bob Lee"])
                        card_name_field = self.driver.find_element(By.ID, "cardName")
                        card_name_field.clear()
                        card_name_field.send_keys(full_name)
                        self.card_name_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"card name failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 10:  # Type card number (16 digits at once)
                if self.current_page == "checkout" and not self.card_number_filled:
                    try:
                        card_number = "".join([str(random.randint(0, 9)) for _ in range(16)])
                        card_number_field = self.driver.find_element(By.ID, "cardNumber")
                        card_number_field.clear()
                        card_number_field.send_keys(card_number)
                        self.card_number_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"card number failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 11:  # Type expiry date (MM/YY format)
                if self.current_page == "checkout" and not self.expiry_filled:
                    try:
                        month = f"{random.randint(1, 12):02d}"
                        year = f"{random.randint(25, 30):02d}"
                        expiry = f"{month}/{year}"
                        expiry_field = self.driver.find_element(By.ID, "expiryDate")
                        expiry_field.clear()
                        expiry_field.send_keys(expiry)
                        self.expiry_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"expiry failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 12:  # Type CVV (3 digits at once)
                if self.current_page == "checkout" and not self.cvv_filled:
                    try:
                        cvv = "".join([str(random.randint(0, 9)) for _ in range(3)])
                        cvv_field = self.driver.find_element(By.ID, "cvv")
                        cvv_field.clear()
                        cvv_field.send_keys(cvv)
                        self.cvv_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"cvv failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 13:  # Type address (full address at once)
                if self.current_page == "checkout" and not self.address_filled:
                    try:
                        address = random.choice(["123 Main St", "42 Wallaby Way", "500 Elm St", "77 Maple Ave"])
                        address_field = self.driver.find_element(By.ID, "address")
                        address_field.clear()
                        address_field.send_keys(address)
                        self.address_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"address failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 14:  # Type city (full city name at once)
                if self.current_page == "checkout" and not self.city_filled:
                    try:
                        city = random.choice(["Toronto", "New York", "Paris", "Tokyo", "Berlin"])
                        city_field = self.driver.find_element(By.ID, "city")
                        city_field.clear()
                        city_field.send_keys(city)
                        self.city_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"city failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 15:  # Type ZIP code (5 digits at once)
                if self.current_page == "checkout" and not self.zip_filled:
                    try:
                        zip_code = "".join([str(random.randint(0, 9)) for _ in range(5)])
                        zip_field = self.driver.find_element(By.ID, "zipCode")
                        zip_field.clear()
                        zip_field.send_keys(zip_code)
                        self.zip_filled = True
                        reward += 3.0
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"zip failed: {e}")
                        reward -= 1.0
                else:
                    reward -= 0.5
                    
            elif action == 16:  # Complete payment
                if self.current_page == "checkout" and self._all_payment_fields_filled():
                    try:
                        pay_btn = self.driver.find_element(By.ID, "payBtn")
                        pay_btn.click()
                        time.sleep(0.5)
                        message = self.driver.find_element(By.ID, "message").text
                        if "successful" in message.lower():
                            self.payment_completed = True
                            reward += 100.0
                            terminated = True
                        else:
                            reward -= 10.0
                    except Exception as e:
                        print(f"payment click/check failed: {e}")
                        reward -= 5.0
                else:
                    reward -= 2.0

            time.sleep(0.01)

        except WebDriverException as e:
            print(f"WebDriver exception in action {action}: {e}")
            self.driver_alive = False
            reward -= 5.0
            terminated = True  # End episode if driver fails
        except Exception as e:
            print(f"Action {action} failed: {e}")
            reward -= 2.0

        # Update cart count
        self.items_in_cart = self._count_cart_items()

        # Reward shaping
        if self.is_logged_in and self.current_page == "store":
            if self.items_in_cart >= self.target_items:
                reward += 0.75

        if self.current_page == "checkout":
            payment_progress = sum([
                self.card_name_filled, self.card_number_filled,
                self.expiry_filled, self.cvv_filled,
                self.address_filled, self.city_filled, self.zip_filled
            ]) / 7.0
            reward += payment_progress * 0.8

        if self.current_page == "store" and (self.steps % 10 == 0):
            reward -= 0.5

        reward -= 0.02

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _click_add_to_cart(self, button_index: int) -> bool:
        """Click the nth 'Add to Cart' button."""
        try:
            buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Add to Cart')]")
            if len(buttons) > button_index:
                buttons[button_index].click()
                return True
        except Exception as e:
            print(f"_click_add_to_cart error: {e}")
        return False

    def _count_cart_items(self) -> int:
        """Count items in cart."""
        try:
            if self.current_page == "store":
                cart_items = self.driver.find_elements(By.CLASS_NAME, "cart-item")
                return min(len(cart_items), 5)
            elif self.current_page == "checkout":
                cart_items = self.driver.find_elements(By.XPATH, "//div[@id='checkoutCart']//div[@class='cart-item']")
                return min(len(cart_items), 5)
        except Exception:
            pass
        return 0

    def _all_payment_fields_filled(self) -> bool:
        """Check if all payment fields are filled."""
        return all([
            self.card_name_filled, self.card_number_filled,
            self.expiry_filled, self.cvv_filled,
            self.address_filled, self.city_filled, self.zip_filled
        ])

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        card_fields_progress = sum([
            self.card_name_filled, self.card_number_filled,
            self.expiry_filled, self.cvv_filled
        ]) / 4.0
        
        address_fields_progress = sum([
            self.address_filled, self.city_filled, self.zip_filled
        ]) / 3.0

        page_encoding = {
            "login": 0.0,
            "store": 0.5,
            "checkout": 1.0
        }

        obs = np.array([
            float(self.is_logged_in),
            min(self.items_in_cart / 5.0, 1.0),
            float(self.current_page == "store"),
            float(self.current_page == "checkout"),
            float(self.username_filled and self.password_filled),
            card_fields_progress,
            address_fields_progress,
            self.steps / max(1, self.max_steps),
            float(self.payment_completed),
            page_encoding.get(self.current_page, 0.0)
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "steps": self.steps,
            "is_logged_in": self.is_logged_in,
            "items_in_cart": self.items_in_cart,
            "current_page": self.current_page,
            "payment_completed": self.payment_completed,
            "login_progress": float(self.username_filled and self.password_filled),
            "payment_progress": float(self._all_payment_fields_filled()),
            "driver_alive": self.driver_alive
        }

    def render(self):
        """Rendering handled by browser."""
        pass

    def close(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        self.driver_alive = False