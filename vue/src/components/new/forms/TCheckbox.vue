<template>
  <div class="t-checkbox" :class="{ 't-checkbox--disabled': isDisabled }">
    <input
      v-model="checked"
      class="t-checkbox__input"
      :checked="checked ? 'checked' : ''"
      type="checkbox"
      :name="parse"
      :data-reverse="reverse"
      :disabled="isDisabled"
      @change="change"
    />
    <span class="t-checkbox__switch"></span>
  </div>
</template>

<script>
export default {
  name: 't-checkbox-new',
  props: {
    value: Boolean,
    name: String,
    parse: String,
    reverse: Boolean,
    disabled: [Boolean, Array],
  },
  data: () => ({
    checked: false,
  }),
  computed: {
    isDisabled() {
      if (Array.isArray(this.disabled)) {
        return !!this.disabled.includes(this.name);
      } else {
        return this.disabled;
      }
    },
  },
  methods: {
    send(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });
      this.$emit('parse', { parse: this.parse, value });
    },
    label() {
      if (!this.isDisabled) {
        this.checked = !this.checked;
        this.send(this.checked);
      }
    },
    change(e) {
      this.send(e.target.checked);
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
  },
  created() {
    this.checked = this.value;
  },
};
</script>

<style lang="scss" scoped>
.t-checkbox {
  width: 35px;
  height: 20px;
  position: relative;
  &--disabled {
    opacity: 0.3;
  }
  &__input {
    width: 100%;
    height: 100%;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    opacity: 0;
    cursor: pointer;
    &:checked + span:before {
      transform: translateX(15px);
    }
    &:checked + span {
      background-color: #65b9f4;
    }
  }
  &__switch {
    background-color: #6C7883;
    display: block;
    position: relative;
    height: 100%;
    border-radius: 20px;
    cursor: pointer;
    &:before {
      background-color: #0E1621;
      display: block;
      content: '';
      height: 16px;
      width: 16px;
      position: absolute;
      left: 2px;
      top: 2px;
      border-radius: 50%;
      transition: 0.2s;
    }
  }
}
</style>
