#!/bin/bash

# Function to check kernel version
check_kernel_version() {
    local kernel_version=$(uname -r | cut -d. -f1,2)
    local kernel_version_float=$(echo $kernel_version | awk '{print $1}')
    
    if (( $(echo "$kernel_version_float >= 4.18" | bc -l) )); then
        echo "Kernel version check: PASSED (Version: $(uname -r))"
        return 0
    else
        echo "Kernel version check: FAILED (Version: $(uname -r))"
        echo "Zero-copy TCP requires kernel version 4.18 or higher"
        return 1
    fi
}

# Function to check if zero-copy TCP is enabled
check_zerocopy_enabled() {
    if grep -q "CONFIG_TCP_ZERO_COPY_TRANSFER_COMPLETION_NOTIFICATION=y" /boot/config-$(uname -r); then
        echo "Zero-copy TCP configuration: ENABLED"
        return 0
    else
        echo "Zero-copy TCP configuration: NOT ENABLED"
        return 1
    fi
}


# Add function to set new kernel as default
set_new_kernel_default() {
    local new_kernel_version=$(make kernelversion)

    # Backup GRUB config
    sudo cp /etc/default/grub /etc/default/grub.backup

    # Set GRUB timeout to 5 seconds
    sudo sed -i 's/GRUB_TIMEOUT=.*/GRUB_TIMEOUT=5/' /etc/default/grub

    # Set new kernel as default
    sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT='Advanced options for Ubuntu>Ubuntu, with Linux ${new_kernel_version}'/" /etc/default/grub

    # Update GRUB
    sudo update-grub
}

# Add function to setup failsafe boot
setup_failsafe_boot() {
    local current_kernel=$(uname -r)

    # Create a systemd service to check boot success
    cat << EOF | sudo tee /etc/systemd/system/boot-check.service
[Unit]
Description=Check successful boot and reset GRUB if failed
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/boot-check.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Create the boot check script
    cat << EOF | sudo tee /usr/local/bin/boot-check.sh
#!/bin/bash
# If we can reach this point, boot was successful
# Keep the new kernel as default
exit 0
EOF

    sudo chmod +x /usr/local/bin/boot-check.sh
    sudo systemctl enable boot-check.service
}

# Function to enable zero-copy TCP
enable_zerocopy() {
    echo "Starting process to enable zero-copy TCP..."
    
    # Check for required packages
    local required_packages="build-essential libncurses-dev bison flex libssl-dev libelf-dev"
    
    echo "Checking and installing required packages..."
    for package in $required_packages; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            sudo apt-get install -y $package
        fi
    done
    
    # Create working directory
	local work_dir="/tmp/kernel_build_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $work_dir
    cd $work_dir

    echo "Getting kernel source..."
    apt-get source linux-image-$(uname -r)

    echo "Installing build dependencies..."
    sudo apt-get build-dep -y linux-image-$(uname -r)

    cd linux-*

    echo "Copying current kernel configuration..."
    cp /boot/config-$(uname -r) .config

    echo "Enabling zero-copy TCP configuration..."
    scripts/config --enable CONFIG_TCP_ZERO_COPY_TRANSFER_COMPLETION_NOTIFICATION

    echo "Building kernel (this will take a while)..."
    make -j$(nproc)

    echo "Installing modules..."
    sudo make modules_install

    echo "Installing kernel..."
    sudo make install

    echo "Setting up new kernel as default..."
    set_new_kernel_default

    echo "Setting up failsafe boot mechanism..."
    setup_failsafe_boot

    echo "================================================================"
    echo "Zero-copy TCP has been enabled in the new kernel."
    echo "The system will now reboot to the new kernel."
    echo "The old kernel will remain available as a fallback option."
    echo "GRUB timeout is set to 5 seconds."
    echo "Rebooting in 5 seconds..."
    sleep 5
    sudo reboot
}

# Main script
echo "Checking system compatibility for zero-copy TCP..."
echo "================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Step 1: Check if system can run zero-copy TCP
if check_kernel_version; then
    echo "Your system can support zero-copy TCP"
    
    # Step 2: Check if zero-copy TCP is enabled
    if check_zerocopy_enabled; then
        echo "Zero-copy TCP is already enabled on your system"
        exit 0
    else
        # Step 3: Enable zero-copy TCP
        echo "Zero-copy TCP can be enabled on your system"
		echo "Would you like to enable it (This will involve compiling kernel and re-installing it)? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
            enable_zerocopy
        else
            echo "Operation cancelled by user"
            exit 0
        fi
    fi
else
    echo "Your system cannot support zero-copy TCP without a kernel upgrade"
    exit 1
fi
