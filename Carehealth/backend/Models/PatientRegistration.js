const mongoose = require('mongoose');
const validator = require('validator');

const patientSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    age: {
        type: Number,
        required: true,
        min: [0, 'Age cannot be negative'], // Ensures age is not negative
    },
    gender: {
        type: String,
        required: true,
    },
    contact: {
        type: String,
        required: true,
        match: [/^[6-9]\d{9}$/, 'Please enter a valid contact number'], // Ensures the number is 10 digits and starts with 9, 8, 7, or 6
    },
    email: {
        type: String,
        required: true,
        unique: true, // Ensures the email is unique
        validate: {
            validator: function(v) {
                return validator.isEmail(v); // Uses validator library to check for email format
            },
            message: props => `${props.value} is not a valid email!`
        }
    },
    password: {
        type: String,
        required: true,
    },
    confirm_password: {
        type: String,
        required: true
    },
    bloodGroup: {
        type: String,
        enum: ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'],
    },
    address: {
        type: String,
    },
    userRole: {
        type: String,
        required: true,
        default: 'patient' 
    },
}, {
    timestamps: true, // Adds createdAt and updatedAt fields
});

const Patient = mongoose.model('Patient', patientSchema);

module.exports = Patient;
