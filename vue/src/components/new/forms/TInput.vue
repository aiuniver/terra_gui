<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" @click="$el.getElementsByTagName('input')[0].focus()">
      <slot>{{ label }}</slot>
    </label>

    <input
      v-model="input"
      :class="['t-field__input', { small: small }, { 't-field__error': error }]"
      :type="type"
      :name="name || parse"
      :value="value"
      :disabled="disabled"
      :data-degree="degree"
      :autocomplete="'off'"
      @blur="change"
      @focus="focus"
      @mouseover="hover = true"
      @mouseleave="hover = false"
    />
    <div v-show="error && hover" :class="['t-field__hint', { 't-inline__hint': inline }]">
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-input',
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
      type: [String, Number, Array],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    error: String,
    degree: Number, // for serialize
    update: Object,
  },
  data: () => ({
    isChange: false,
    hover: false,
  }),
  computed: {
    input: {
      set(value) {
        // console.log(value);
        this.$emit('input', value);
        this.isChange = true;
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    focus(e) {
      // console.log(e);
      this.$emit('focus', e);
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
    change(e) {
      // if (+e.target.value > 99 && this.name === 'classes') e.target.value = '99'
      let value = e.target.value;
      if (this.isChange && value !== '') {
        // console.log(e);
        value = this.type === 'number' ? +value : value;
        this.$emit('change', { name: this.name, value });
        this.$emit('parse', { name: this.name, parse: this.parse, value });
      }
      this.isChange = false;
    },
  },
  created() {
    this.input = this.value;
  },
  watch: {
    update(obj) {
      console.log(obj);
      if (obj[this.name]) {
        console.log('ok');
        this.$el.getElementsByTagName('input')[0].value = obj[this.name];
        this.$nextTick(() => {
          this.input = obj[this.name];
        });
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 10px;
  position: relative;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1;
    font-size: 0.75rem;
    // text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #65B9F4;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      background: rgba(101, 185, 244, 0.15);
    }
  }
  &__error {
    border-color: #ca5035;
    color: #ca5035;
  }
  &__input.small {
    height: 24px;
    width: 24px;
    font-size: 12px;
    line-height: 24px;
    padding: 0 3px;
  }
  &__hint {
    user-select: none;
    position: absolute;
    height: 22px;
    display: flex;
    align-items: center;
    padding: 0 5px 0 5px;
    top: 65px;
    background-color: #ca5035;
    color: #fff;
    border-radius: 4px;
    z-index: 5;
    // display: none;
    span {
      font-style: normal;
      font-weight: normal;
      font-size: 9px;
      line-height: 12px;
    }
    // &--hover {
    //   display: flex;
    // }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  .t-field__label {
    // width: 150px;
    // max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  .t-field__input {
    padding: 0 5px;
    height: 30px;
    font-size: 14px;
    line-height: 24px;
    width: 150px;
    & .small {
      height: 24px;
      // width: 24px;
      font-size: 12px;
      line-height: 24px;
      padding: 0 3px;
    }
  }
  &__hint {
    top: 25px;
  }
}

input[type="number"]:focus {
  border-color: #65B9F4;
}
</style>
