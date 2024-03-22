# train a generative adversarial network on a one-dimensional data
from sklearn.preprocessing import MinMaxScaler
from numpy import zeros
from numpy import ones
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from pandas import read_csv

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
		
# define the standalone generator model
def define_generator(latent_dim, n_outputs=2): 
	model = Sequential()
	model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(n_outputs, activation='linear'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# generate n real samples with class labels
def generate_real_samples(data, n):
	X = data
	# generate class labels
	y = ones((n, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(data):
	x_input = data
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, data, n):
	# generate points in latent space
	x_input = generate_latent_points(data)
	# predict outputs
	X = generator.predict(x_input, verbose=0)
	# create class labels
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, data):
	# prepare real samples
	x_real, y_real = generate_real_samples(data, len(data))
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(generator, data, len(data))
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print("Epoch: "+str(epoch), "Acc_Real: "+str(acc_real), "Acc_Fake: "+str(acc_fake))
	# plot real and fake data points
	for i in range(0, x_real.shape[1]):
		pyplot.plot(x_real[:, i], color='red', label='Real')
		pyplot.plot(x_fake[:, i], color='blue', label='Fake')
		pyplot.legend()
		pyplot.title('Attribute: '+str(i+1))
		pyplot.show()

# train the generator and discriminator
def train(g_model, d_model, gan_model, data, n_epochs=10, n_eval=1):
	# manually enumerate epochs
	print("[INFO] GAN is training ...")
	for i in range(n_epochs):
		# prepare real samples
		x_real, y_real = generate_real_samples(data, len(data))
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, data, len(data))
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(data)
		# create inverted labels for the fake samples
		y_gan = ones((len(data), 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, data)
			print("[INFO] GAN is training ...")

# define dataset
data = read_csv('household_power_consumption_months.csv', header=0, index_col=0)
data = data.values

# transofrm data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# size of the latent space
latent_dim = data.shape[1]
n_inputs = data.shape[1]
n_outputs = data.shape[1]

# create the discriminator
discriminator = define_discriminator(n_inputs)
# create the generator
generator = define_generator(latent_dim, n_outputs)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, data, n_epochs=1000, n_eval=1000)
print("[INFO] GAN is trained ...")
