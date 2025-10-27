const products = [
  { id: 1, name: "Laptop", price: 999 },
  { id: 2, name: "Headphones", price: 199 },
  { id: 3, name: "Mouse", price: 49 },
];

let cart = [];

const productsDiv = document.getElementById("products");
const cartDiv = document.getElementById("cart");
const message = document.getElementById("message");
const checkoutBtn = document.getElementById("checkoutBtn");

function renderProducts() {
  productsDiv.innerHTML = "";
  products.forEach(p => {
    const el = document.createElement("div");
    el.className = "product";
    el.innerHTML = `
      <span>${p.name} - $${p.price}</span>
      <button onclick="addToCart(${p.id})">Add to Cart</button>
    `;
    productsDiv.appendChild(el);
  });
}

function renderCart() {
  cartDiv.innerHTML = "";
  if (cart.length === 0) {
    cartDiv.innerHTML = "<p>Cart is empty.</p>";
    return;
  }
  cart.forEach(item => {
    const el = document.createElement("div");
    el.className = "cart-item";
    el.innerHTML = `
      <span>${item.name} - $${item.price}</span>
      <button onclick="removeFromCart(${item.id})">Remove</button>
    `;
    cartDiv.appendChild(el);
  });
}

function addToCart(id) {
  const product = products.find(p => p.id === id);
  if (product) cart.push(product);
  renderCart();
  message.textContent = "";
}

function removeFromCart(id) {
  cart = cart.filter(item => item.id !== id);
  renderCart();
}

checkoutBtn.onclick = () => {
  if (cart.length === 0) {
    message.textContent = "? Cannot checkout an empty cart!";
    return;
  }
  message.textContent = "? Payment successful! Thank you for your purchase.";
  cart = [];
  renderCart();
};

renderProducts();
renderCart();
