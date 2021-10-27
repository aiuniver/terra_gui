<template>
  <div class="d-checkbox" :class="{ 'd-checkbox--disabled': isDisabled }">
    <input
      v-model="checked"
      class="d-checkbox__input"
      :checked="checked ? 'checked' : ''"
      type="checkbox"
      :name="name"
      :disabled="isDisabled"
      @change="change"
    />
    <span class="d-checkbox__switch"></span>
  </div>
</template>

<script>
import fields from '../../mixins/fields';
export default {
  name: 'd-checkbox',
  mixins: [fields],
  props: {
    value: Boolean,
    name: String,
    reverse: Boolean,
    disabled: [Boolean, Array],
  },
  data: () => ({
    checked: false,
  }),
  computed: {
  },
  methods: {
    send(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });
    },
    label() {
      if (!this.isDisabled) {
        this.checked = !this.checked;
        this.send(this.checked);
      }
    },
    change(e) {
      this.send(e.target.checked);
    },
  },
  created() {
    this.checked = this.value;
  },
};
</script>

<style lang="scss" scoped>
.d-checkbox {
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
      transform: translateX(14px);
    }
    &:checked + span {
      background-color: #6C7883;
    }
  }
  &__switch {
    display: block;
    position: relative;
    height: 100%;
    border-radius: 20px;
    background-color: #65B9F4;
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
