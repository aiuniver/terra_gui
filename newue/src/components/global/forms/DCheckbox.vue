<template>
  <div class="t-field t-inline" style="margin: 0;">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div :class="['t-field__switch', { 't-field__switch--checked': checked, 't-field__switch--disabled': disabled }]">
      <input
        :id="parse"
        class="t-field__input"
        :checked="checked ? 'checked' : ''"
        type="checkbox"
        :value="checked"
        :name="parse"
        :disabled="disabled"
        @change="change"
      />
      <span></span>
    </div>
  </div>
</template>

<script>
import { bus } from '@/main';
export default {
  name: 'd-checkbox',
  props: {
    label: {
      type: String,
      default: '',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [Boolean],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
    event: {
      type: Array,
      default: () => [],
    },
    disabled: Boolean
  },
  data: () => ({
    checked: null,
  }),
  methods: {
    change(e) {
      const value = e.target.checked;
      this.checked = value
      this.$emit('input', value)
      this.$emit('change', { name: this.name, value, parse: this.parse });
      bus.$emit('change', { event: this.name, value });
    }
  },
  created() {
    this.checked = this.value;
    if (this.event.length) {
      bus.$on('change', ({ event }) => {
        if (this.event.includes(event)) {
          this.checked = false;
        }
      });
    }
  },
  destroyed() {
    if (this.event.length) {
      bus.$off();
      console.log('destroyed', this.name);
    }
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  &__label {
    width: 150px;
    max-width: 330px;
    padding: 0 10px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    &:focus {
      border-color: #fff;
    }
    &:checked + span:before {
      transform: translateX(15px);
      background-color: #0E1621;
    }
  }
  &__switch {
    width: 35px;
    height: 20px;
    position: relative;
    background: #6C7883;
    border-radius: 20px;
    &--checked {
      background: #65B9F4;
    }
    &--disabled {
      opacity: 0.5;
      input {
        cursor: default;
      }
    }
    span {
      display: block;
      position: relative;
      height: 100%;
      transition: 0.2s;
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
}

.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;

  > label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  > input {
    width: 100%;
    height: 100%;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    opacity: 0;
    cursor: pointer;
  }
}
</style>
