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
  width: 26px;
  height: 14px;
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
      transform: translateX(12px);
      background-color: #65b9f4;
    }
  }
  &__switch {
    background-color: #242f3d;
    display: block;
    position: relative;
    height: 100%;
    border: 1px solid #6c7883;
    border-radius: 4px;
    cursor: pointer;
    &:before {
      background-color: #6c7883;
      display: block;
      content: '';
      height: 10px;
      width: 10px;
      position: absolute;
      left: 1px;
      top: 1px;
      border-radius: 2px;
      transition: 0.2s;
    }
  }
}
</style>
