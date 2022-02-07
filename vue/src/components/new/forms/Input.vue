<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <input
      v-model="input"
      :class="['t-field__input', { 't-field__input--error': error ? true : false , 't-field__input--disabled': disabled }]"
      :id="parse"
      :type="type"
      :name="parse"
      :value="value"
      autocomplete="off"
      @blur="change"
      @input="enter"
      :disabled="disabled"
    />
  </div>
</template>

<script>
import { debounce } from '@/utils/core/utils';
export default {
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    error: String,
  },
  data: () => ({
    isChange: false,
    debounce: null,
    temp: ''
  }),
  computed: {
    input: {
      set(value) {
        this.$emit('input', value);
        this.temp = value
        this.isChange = true;
      },
      get() {
        return this.value;
      },
    },
  },
  mounted() {
    this.debounce = debounce(e => {
      this.change(e);
    }, 500);
  },
  beforeDestroy() {
    if (this.isChange) {
      let value = this.temp.trim();
      this.$emit('change', { name: this.name, value});
      this.isChange = false;
    }
  },
  methods: {
    enter(e) {
      this.debounce(e);
    },
    change(e) {
      // console.log(e);
      if (this.isChange) {
        let value = e.target.value.trim();
        // console.log(typeof value);
        if (value !== '') {
          value = this.type === 'number' ? +value : value;
        } else {
          value = null;
        }
        this.$emit('change', { name: this.name, value });
        this.isChange = false;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 10px;
  &__label {
    width: 150px;
    max-width: 130px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 1rem;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #65B9F4;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      background: rgba(101, 185, 244, 0.15);
    }
    &--error {
      border-color: #f00;
    }
    &--disabled {
      background: rgba(36, 47, 61, 0.5);
      opacity: 0.5;
      border: 1px solid #242F3D;
    }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
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
    height: 30px;
    font-size: 1rem;
    padding: 0 5px;
    line-height: 24px;
    width: 150px;
  }
}

input:focus {
  border-color: #65B9F4;
}
</style>
