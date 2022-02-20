<template>
  <div :class="['t-field', { 't-color': inline }]">
    <div class="t-field__wrapper">
      <label class="t-field__label" @click="$el.getElementsByTagName('input')[0].focus()">
        <slot>{{ label }}</slot>
      </label>

      <div class="t-field__content" v-click-outside="outside">
        <div v-if="pickerShow" class="t-field__color-picker">
          <ColorPicker
            v-model="input"
            :width="size.x"
            :height="size.y"
            :disabled="false"
            startColor="#ff0000"
            @color-change="$emit('change', $event)"
          ></ColorPicker>
        </div>
        <input
          v-model="input"
          :class="['t-field__input', { small: small }, { 't-field__error': error }]"
          :type="type"
          :name="name || parse"
          :data-degree="degree"
          :autocomplete="'off'"
          :disabled="disabled"
          @blur="change"
          @focus="focus"
        />
        <div class="t-field__box" :style="{ background: input }" @click="click"></div>
      </div>
    </div>
  </div>
</template>

<script>
import ColorPicker from './ColorPicker.vue';

export default {
  name: 't-color',
  components: {
    ColorPicker,
  },
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
    small: Boolean,
    error: String,
    degree: Number, // for serialize
    size: {
      type: Object,
      default: () => ({ x: 100, y: 100 })
    }
  },
  data: () => ({
    isChange: false,
    pickerShow: false,
    input: '',
  }),
  computed: {
    // input: {
    //   set(value) {
    //     this.$emit('input', value);
    //     this.isChange = true;
    //   },
    //   get() {
    //     return this.value;
    //   },
    // },
  },
  methods: {
    click () {
      if (!this.disabled) [
        this.pickerShow = true
      ]
    },
    outside() {
      if (this.pickerShow) {
        this.pickerShow = false;
      }
    },
    focus(e) {
      this.$emit('focus', e);
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
    change(e) {
      if (this.isChange) {
        let value = e.target.value;
        value = /^#([0-9A-F]{3}){1,2}$/i.test(value) ? value : 'err';
        this.$emit('change', { name: this.name, value });
        this.isChange = false;
      }
    },
    onColorChange(color) {
      this.input = color.toUpperCase();
      // console.log("asdasdas     ", this.input)
    },
  },
  created() {
    this.input = this.value
  }
};
</script>

<style lang="scss" scoped>
.t-field {
  position: relative;
  &__content {
    position: relative;
  }
  &__color-picker ~ .t-field__input {
      border-radius: 0 0 4px 4px;
      border-color: #fff;
    }
  &__box {
    background: #59b9ff;
    width: 20px;
    height: 20px;
    position: absolute;
    left: 2px;
    top: 50%;
    transform: translateY(-50%);
    border-radius: 2px;
  }
  &__color-picker {
    position: absolute;
    bottom: 13px;
    background-color: #1b2a3f;
    border-radius: 4px 4px 0 0;
    border: 1px solid #fff;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 10px;
    width: 100px;
    // padding: 10px;
  }
  // margin-bottom: 20px;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1.5;
    font-size: 0.75rem;
    // text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    background: #242f3d;
    height: 42px;
    padding: 0 10px 0 27px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #6c7883;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
  }
  &__error {
    border-color: #b53b3b;
  }
  &__input.small {
    height: 24px;
  }
  &__wrapper {
    display: flex;
    flex-direction: row-reverse;
    justify-content: flex-end;
    align-items: center;
    -webkit-box-pack: end;
    margin-bottom: 10px;
    width: 100%;
    .t-field__label {
      width: 100%;
      padding: 0 10px;
      text-align: left;
      color: #a7bed3;
      display: block;
      margin: 0;
      line-height: 1;
      font-size: 0.75rem;
    }
    .t-field__input {
      height: 24px;
      font-size: 12px;
      line-height: 24px;
      width: 100px;
    }
  }
}
.t-color {
  display: flex;
  flex-direction: column;
  align-items: center;
}
slide-fade-enter-active {
  transition: all 0.3s ease;
}
.slide-fade-leave-active {
  transition: all 0.8s cubic-bezier(1, 0.5, 0.8, 1);
}
.slide-fade-enter,
.slide-fade-leave-to {
  transform: translateX(10px);
  opacity: 0;
}
</style>
