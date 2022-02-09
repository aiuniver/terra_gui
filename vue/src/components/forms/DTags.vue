<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div class="d-tags">
      <button class="d-tags__add mr-2" type="button">
        <i class="d-tags__add--icon t-icon icon-tag-plus" @click="create"></i>
        <input type="text" class="d-tags__add--input" :disabled="tags.length >= 3" :placeholder="'Добавить'" @keypress.enter.prevent="create" />
      </button>
      <template v-for="({ value }, i) in tags">
        <div class="d-tags__item mr-2" :key="'tag_' + i">
          <input
            :value="value"
            :data-index="i"
            name="[tags][][name]"
            type="text"
            class="d-tags__input"
            :style="{ width: (value.length + 1) * 8 <= 90 ? (value.length + 1) * 8 + 'px' : '90px' }"
            autocomplete="off"
            @input="change"
            @blur="blur"
          />
          <i class="d-tags__remove--icon t-icon icon-tag-plus" @click="removeTag(i)"></i>
          <!-- <div class="d-tags__item"></div> -->
        </div>

        <!-- <span class="d-tags__add--span" :key="'span_' + i">{{ value }}</span> -->
      </template>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-input',
  props: {
    label: {
      type: String,
      default: 'Теги',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: Array,
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
  },
  data: () => ({
    // tags: [],
  }),
  computed: {
    tags: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    removeTag(index) {
      this.tags = this.tags.filter((item, i) => i !== +index);
    },
    create() {
      const el = this.$el.getElementsByClassName('d-tags__add--input')?.[0];
      console.log(el.value);
      if (el.value && el.value.length >= 3 && this.tags.length <= 3) {
        this.tags.push({ value: el.value });
        this.tags = [...this.tags];
        el.value = '';
      }
    },
    change(e) {
      const index = e.target.dataset.index;
      console.log(index);
      if (e.target.value.length >= 3) {
        this.tags[+index].value = e.target.value;
      }
    },
    blur(e) {
      const index = e.target.dataset.index;
      if (e.target.value.length <= 2) {
        this.tags = this.tags.filter((item, i) => i !== +index);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  // margin-bottom: 20px;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border-color: #6c7883;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
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
    width: 150px;
    max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  > input {
    height: 40px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
}

.d-tags {
  display: flex;
  max-width: 400px;
  &__remove {
    &--icon {
      position: absolute;
      height: 12px;
      width: 12px;
      top: 40%;
      right: 2px;
      cursor: pointer;
      transform: rotate(45deg);
    }
  }
  &__input {
    padding: 0 10px 0 5px;
  }
  &__item {
    position: relative;
    margin-right: 5px;
    // margin-left: 10px;
    // border: none;
    // padding: 2px;
    width: 60px;
    height: 40px;
    color: #a7bed3;
    font-size: 12px;
    font-weight: normal;
    line-height: 24px;
    // margin-left: 8px;
    text-align: center;
  }
  &__error {
    border-color: red;
  }
  &__add {
    background: #242f3d;
    height: 40px;
    width: 90px;
    padding: 2px 4px;
    box-shadow: none;
    border-color: #6c7883;
    color: #a7bed3;
    display: flex;
    align-items: center;
    &--icon {
      height: 16px;
      width: 16px;
    }
    &--input {
      height: 40px;
      color: #a7bed3;
      border: none;
      background-color: transparent;
      font-size: 12px;
      padding: 0;
      font-weight: normal;
      line-height: 24px;
      margin-left: 8px;
    }
    &--span {
      background: #242f3d;
      height: 40px;
      padding: 2px 8px;
      box-shadow: none;
      color: #a7bed3;
      display: flex;
      align-items: center;
      border: 1px solid #6c7883;
      border-radius: 4px;
      margin-left: 10px;
      font-size: 12px;
      font-weight: normal;
      line-height: 24px;
    }
  }
}
</style>
