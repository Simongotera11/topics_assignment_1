1. Run `python serv_app.py` in `\static` directory and make sure web app is running in `http://localhost:8080`

2. Make sure to install all depedencies.

3. To train the model headless mode (recommended for speed) run `python train_ecommerce.py --timesteps 10000 --headless --ent_coef 0.05`

4. Train the model with visible browser (watch it learn) run `python train_ecommerce.py --timesteps 50000 --ent_coef 0.05`

5. To test the trained model `python eval_ecommerce.py --model_path models/best_model/best_model.zip --episodes 5 --render`.

