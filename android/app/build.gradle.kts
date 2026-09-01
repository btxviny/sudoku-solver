plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sudokusolver"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.sudokusolver"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        // Only the Pixel's ABI. Dropping the others keeps OpenCV's native
        // libraries from tripling the APK for devices this will never run on.
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    androidResources {
        // TFLite models are memory-mapped straight out of the APK, which only
        // works if they were stored uncompressed.
        noCompress += listOf("tflite")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // LiteRT (formerly TensorFlow Lite) plus the GPU delegate.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")

    // OpenCV: add the Android SDK module and uncomment. See android/README.md --
    // it is a manual step because OpenCV publishes no first-party Maven artifact.
    implementation("org.opencv:opencv:4.10.0")

    testImplementation("junit:junit:4.13.2")
}
