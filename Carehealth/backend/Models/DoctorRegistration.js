const mongoose = require('mongoose');
const validator = require('validator');

const doctorSchema = new mongoose.Schema({
    name: {
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
    specialization: {
        type: String,
        required: true,
    },
    experience: {
        type: Number,
        required: true,
        min: [0, 'Experience cannot be negative'], // Ensures experience is not negative
    },
    qualification: {
        type: String,
        required: true,
    },
    userRole: {
        type: String,
        required: true,
        default: 'doctor' 
    },
}, {
    timestamps: true, // Adds createdAt and updatedAt fields
});

const Doctor = mongoose.model('Doctor', doctorSchema);

module.exports = Doctor;
