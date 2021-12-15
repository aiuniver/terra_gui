<template>
  <span
    class="vue-ip"
    :class="{ 'show-port': portCopy !== false, 'material-theme': theme === 'material', active: active, valid: valid }"
  >
    <div class="segment" v-for="(segment, index) in ipCopy" :key="index">
      <input
        type="number"
        v-model="ipCopy[index]"
        placeholder="___"
        maxlength="3"
        @paste="paste($event)"
        @keydown="ipKeydown($event, index)"
        @focus="ipFocus(index)"
        @blur="blur"
        @input="input"
        ref="ipSegment"
      />
    </div>
  </span>
</template>

<script>
export default {
  props: {
    onChange: Function,
    ip: {
      required: true,
      type: String,
    },
    port: {
      type: [String, Number, Boolean],
      default: false,
    },
    placeholder: {
      type: [Boolean],
      default: false,
    },
    theme: {
      type: [String, Boolean],
      default: false,
    },
  },
  data() {
    return {
      ipCopy: ['', '', '', ''],
      portCopy: null,
      valid: false,
      active: false,
    };
  },
  beforeMount() {
    this.copyValue(this.ip, this.port);
  },
  watch: {
    ip(newIp) {
      this.copyValue(newIp, this.port);
    },
    port(newPort) {
      this.copyValue(this.ip, newPort);
    },
  },
  methods: {
    input(e) {
      if (!e.target.value) e.target.value = '';
    },
    placeholderPos(segment) {
      if (!this.placeholder) return '';

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
    ipFocus(index) {
      this.active = true;
      this.ipCopy[index] = '';
      this.changed();
    },
    clearAll() {
      this.ipCopy = ['', '', '', ''];
      this.portCopy = null;
      this.valid = false;
    },
    blur() {
      this.active = false;
    },
    portFocus() {
      this.active = true;
      this.portCopy = null;
      this.changed();
    },
    paste(event) {
      this.$refs.ipSegment[0].focus();
      let pasteText = event.clipboardData.getData('text/plain');
      let portPos = pasteText.indexOf(':');
      if (this.port === false) {
        console.warn(
          'A IP address with a port has been entered but this module has no port attribute. Please enable it update the port.'
        );
        this.clearAll();
        let ipAndPort = pasteText.split(':');
        this.copyValue(ipAndPort[0], false);
        this.$refs.ipSegment[0].blur();

        return;
      }
      switch (portPos) {
        case -1:
          this.copyValue(pasteText, null);
          this.changed();
          this.$refs.ipSegment[0].blur();

          break;
        default:
          let ipAndPort = pasteText.split(':'); // eslint-disable-line
          this.copyValue(ipAndPort[0], ipAndPort[1]);
          this.changed();
          this.$refs.ipSegment[0].blur();

          break;
      }
    },
    ipKeydown(event, index) {
      let keyCode = event.keyCode || event.which;
      if (keyCode === 8 || keyCode === 37) {
        if (this.ipCopy[index].length === 0 && this.ipCopy[index - 1] !== undefined)
          this.$refs.ipSegment[index - 1].focus();
      }

      setTimeout(() => {
        if (this.ipCopy[index] === '0') this.moveToNextIpSegment(index, false);
        else this.moveToNextIpSegment(index);
        this.changed();
      });
    },
    moveToNextIpSegment(index, ifOverThree = true) {
      if (ifOverThree) {
        if (this.ipCopy[index].length >= 3 && this.ipCopy[index + 1] !== undefined)
          this.$refs.ipSegment[index + 1].focus();
      } else if (!ifOverThree) {
        if (this.ipCopy[index + 1] !== undefined) this.$refs.ipSegment[index + 1].focus();
      }
    },
    changed(ip = this.ipCopy, port = this.portCopy) {
      let ipLocal = this.arrayToIp(ip);
      this.onChange(ipLocal, port, this.validateIP(ip));
    },
    copyValue(ip, port) {
      if (ip) this.ipToArray(ip);
      this.portCopy = port;
      this.valid = this.validateIP(this.ipCopy);
      this.changed();
    },
    ipToArray(ip) {
      let segments = [];
      ip.split('.').map(segment => {
        if (isNaN(segment) || segment < 0 || segment > 255) segment = 255;
        segments.push(segment);
      });
      if (segments.length !== 4) {
        console.error('Not valid, so clearing ip', segments);
        this.clearAll();
      } else this.ipCopy = segments;
    },
    arrayToIp(ipArray) {
      return ipArray.join('.');
    },
    validateIP(ip) {
      let ipCheck = this.arrayToIp(ip);
      return /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/.test(
        ipCheck
      );
    },
  },
};
</script>


<style lang="scss" scoped>
.vue-ip {
  display: flex;
  border: 1px solid #6c7883;
  border-radius: 4px;
  padding: 0 10px;
  background: #242f3d;
  height: 42px;
  &:focus-within {
    border-color: #fff;
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
    input::placeholder {
      color: #666;
    }
    &::after {
      content: '.';
      position: absolute;
      right: -2px;
      bottom: 6px;
      font-weight: bold;
      color: #666;
    }
    &:last-child::after {
      content: '';
    }
  }
}
</style>