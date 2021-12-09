<template>
    <span class="vue-ip" :class="{'show-port' : portCopy !== false, 'material-theme': theme === 'material', 'active': active, 'valid': valid}">
        <div class="segment" v-for="(segment, index) in ipCopy" :key="index">
            <input type="number" v-model="ipCopy[index]" 
            :placeholder="placeholderPos(index)"
            maxlength="3"
            @paste="paste($event)"
            @keydown="ipKeydown($event, index)" 
            @focus="ipFocus(index)"
            @blur="blur"
            @input="input"
            ref="ipSegment" />
        </div>
    </span>
</template>

<style lang="scss" scoped>
.vue-ip {
    display: flex;
    border: 1px solid #242f3d;
    border-radius: 4px;
    padding: 0 10px;
    background: #1b2531;
    height: 40px;
    &:hover, &:focus-within {
        background-color: #65b9f426;
        border-color: #65b9f4;
        box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
    }
    input {
        border: none;
        border-radius: 0;
        background: none;
        padding: 0;
        height: 100%;
        text-align: center;
    }
    .segment {
        position: relative;
        max-width: 35px;
        &::after {
            content: ".";
            position: absolute;
            right: -2px;
            bottom: 6px;
        }
        &:last-child::after {
            content: "";
        }
    }
}
</style>

<script>
    export default {
        props: {
            onChange: Function,
            ip: {
                required: true,
                type: String
            },
            port: {
                type: [String, Number, Boolean],
                default: false
            },
            placeholder: {
                type: [Boolean],
                default: false
            },
            theme: {
                type: [String, Boolean],
                default: false
            }
        },
        data() {
            return {
                ipCopy: ['', '', '', ''],
                portCopy: null,
                valid: false,
                active: false
            }
        },
        beforeMount() {

            // Copy the values over
            this.copyValue(this.ip, this.port);

        },
        watch: {

            /**
             * Watch the IP prop for changes and update internally
             */
            ip(newIp) {
                this.copyValue(newIp, this.port);
            },

            /**
             * Watch the port for changes and update internally
             */
            port(newPort) {
                this.copyValue(this.ip, newPort);
            }

        },
        methods: {
            input(e) {
                if (!e.target.value) e.target.value = ''
            },
            /**
             * Placeholder with dummy IP
             */
            placeholderPos(segment) {

                // No placeholder
                if (!this.placeholder)
                    return '';

                // Dummy IP placeholder
                switch (segment) {
                    case 0:
                        return '192';
                    case 1:
                        return '168';
                    case 2:
                        return '0';
                    case 3:
                        return '1';
                }

            },

            /**
             * On focus clear the current box
             */
            ipFocus(index) {

                this.active = true;

                // Clear it
                this.ipCopy[index] = '';

                // Update the change
                this.changed();

            },

            /**
             * Clear both inputs
             */
            clearAll() {
                this.ipCopy = ['', '', '', ''];
                this.portCopy = null;
                this.valid = false;
            },

            /**
             * Clicked off the IP or port
             */
            blur() {

                this.active = false;

            },

            /**
             * Focus on the port
             */
            portFocus() {

                this.active = true;

                // Clear it
                this.portCopy = null;

                // Update the change
                this.changed();

            },

            /**
             * Paste in an IP (with or without a port)
             */
            paste(event) {

                // Focus on first el
                this.$refs.ipSegment[0].focus();

                // Get clipboard text
                let pasteText = event.clipboardData.getData('text/plain');

                // Check if we have a port or not
                let portPos = pasteText.indexOf(':');

                // If we have ports turned off, remove the port and only update the IP value
                if (this.port === false) {

                    console.warn('A IP address with a port has been entered but this module has no port attribute. Please enable it update the port.');

                    this.clearAll();

                    let ipAndPort = pasteText.split(":");
                    this.copyValue(ipAndPort[0], false);

                    // Blur off input
                    this.$refs.ipSegment[0].blur();

                    return;
                }

                // Based on if we have a port or not
                switch (portPos) {
                    case -1:
                        this.copyValue(pasteText, null);
                        this.changed();

                        // Blur off input
                        this.$refs.ipSegment[0].blur();

                        break;
                    default:
                        let ipAndPort = pasteText.split(":"); // eslint-disable-line
                        this.copyValue(ipAndPort[0], ipAndPort[1]);
                        this.changed();

                        // Blur off input
                        this.$refs.ipSegment[0].blur();

                        break
                }

            },

            /**
             * Run on keydown
             */
            ipKeydown(event, index) {

                let keyCode = event.keyCode || event.which;
                console.log('keycode', keyCode)
                // Return or left on keypad
                if (keyCode === 8 || keyCode === 37) {

                    // If there is nothing within the selected input go the the one before it
                    if (this.ipCopy[index].length === 0 && this.ipCopy[index - 1] !== undefined)
                        this.$refs.ipSegment[index - 1].focus();

                }

                setTimeout(() => {

                    // If its a 0 then always move to the next segment, if not work out if we need to move first
                    if (this.ipCopy[index] === '0')
                        this.moveToNextIpSegment(index, false);
                    else
                        this.moveToNextIpSegment(index);

                    // Update the change
                    this.changed();

                });
            },

            /**
             * Work out if we need to move to the next IP segment or not
             */
            moveToNextIpSegment(index, ifOverThree = true) {

                /**
                 * If there is 3 characters check if there is another segment, if there is focus on it.
                 */
                if (ifOverThree) {

                    if (this.ipCopy[index].length >= 3 && this.ipCopy[index + 1] !== undefined)
                        this.$refs.ipSegment[index + 1].focus();

                } else if (!ifOverThree) {

                    if (this.ipCopy[index + 1] !== undefined)
                        this.$refs.ipSegment[index + 1].focus();

                }

            },

            /**
             * Update the controller with changed IP and port addresses
             */
            changed(ip = this.ipCopy, port = this.portCopy) {
                let ipLocal = this.arrayToIp(ip);
                this.onChange(ipLocal, port, this.validateIP(ip));
            },

            /**
             * Copy prop into local copy
             */
            copyValue(ip, port) {

                if (ip)
                    this.ipToArray(ip);

                // Update the port as long as its a number
                this.portCopy = port;

                // Update if its valid locally
                this.valid = this.validateIP(this.ipCopy);

                // Report right back with if its valid or not
                this.changed();

            },

            /**
             * Convert the IP address string to an array of values
             */
            ipToArray(ip) {

                let segments = [];
                ip.split('.').map(segment => {
                    if (isNaN(segment) || segment < 0 || segment > 255)
                        segment = 255;
                    segments.push(segment);
                });

                // If something is not valid clear it all
                if (segments.length !== 4) {
                    console.error('Not valid, so clearing ip', segments);
                    this.clearAll();
                } else
                    this.ipCopy = segments;

            },

            /**
             * Convert the array of IP segments back to a string
             */
            arrayToIp(ipArray) {
                return ipArray.join(".");
            },

            /**
             * validates the IP address
             *
             * @returns Boolean
             */
            validateIP(ip) {
                let ipCheck = this.arrayToIp(ip);
                return (/^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/.test(ipCheck))
            }

        }
    }
</script>
