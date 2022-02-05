<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" @click="clickLabel">
      <slot>{{ label }}</slot>
    </label>
    <div class="t-field__switch">
      <input
        v-model="checVal"
        :class="['t-field__input', { 't-field__error': error }]"
        :checked="checVal ? 'checked' : ''"
        type="checkbox"
        :value="checVal"
        :name="parse"
        :id="name"
        :data-reverse="reverse"
        @change="change"
        @mouseover="hover = true"
        @mouseleave="hover = false"
      />
      <span></span>
    </div>
    <div v-if="error && hover" class="t-field__hint">
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-checkbox',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    inline: Boolean,
    value: Boolean,
    name: String,
    parse: String,
    error: String,
    reverse: Boolean,
    event: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    checVal: false,
    hover: false,
  }),
  methods: {
    change(e) {
      const value = e.target.checked;
      this.$emit('change', { name: this.name, value });
      this.$emit('parse', { parse: this.parse, value });
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
    clickLabel() {
      this.checVal = !this.checVal;
      this.$emit('change', { name: this.name, value: this.checVal });
      this.$emit('parse', { name: this.name, parse: this.parse, value: this.checVal });
    },
  },
  created() {
    this.checVal = this.value;
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
  position: relative;

  &__label {
    width: 150px;
    max-width: 330px;
    padding-bottom: 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    // white-space: nowrap;
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
    &:checked + span {
      background: #65B9F4;
      &::before {
        transform: translateX(15px);
      }
    }
  }
  &__error {
    border-color: #ca5035;
    color: #ca5035;
  }
  &__switch {
    width: 35px;
    height: 20px;
    position: relative;

    span {
      background-color: #6C7883;
      display: block;
      position: relative;
      height: 100%;
      border-radius: 20px;
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
  &__hint {
    user-select: none;
    position: absolute;
    height: 22px;
    display: flex;
    align-items: center;
    padding: 0 5px 0 5px;
    top: 25px;
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

  > label {
    width: auto;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
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
