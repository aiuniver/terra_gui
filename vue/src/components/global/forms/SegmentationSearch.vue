<template>
  <div class="t-segmentation-search">
    <t-input
      v-model="qty"
      label="Количество классов"
      type="number"
      name="classes"
      min="1"
      max="99"
      size="1"
      maxlength="2"
      inline
      :error="error"
      @focus="error = ''"
      @change="change"
    />
    <div :class="['t-inline']">
      <label class="t-field__label"><slot></slot></label>
      <t-button class="t-field__button" :disabled="disabled" @click.native="getApi" :loading="loading">Найти</t-button>
    </div>
    <template v-for="({ name, color }, i) of items">
      <hr class="t-segmentation-search__hr" :key="'hr_up' + i" />
      <t-input
        :value="name"
        label="Название класса"
        type="text"
        name="classes_names"
        :key="'classes_names_' + i"
        :parse="'classes_names[]'"
        inline
        autocomplete="off"
        @change="change($event, i)"
      />
      <Color :value="color" label="Цвет" :key="'classes_colors_' + i" inline :disabled="true" />
      <hr v-if="items.length === i + 1" class="t-segmentation-search__hr" :key="'hr_' + i" />
    </template>
  </div>
</template>

<script>
import Color from '../../forms/Color.vue';
import blockMain from '@/mixins/datasets/blockMain';
export default {
  name: 't-segmentation-search',
  components: {
    Color,
  },
  mixins: [blockMain],
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
    // error: String,
    id: Number,
  },
  data: () => ({
    loading: false,
    isShow: false,
    items: [],
    classes_names: [],
    classes_colors: [],
    qty: 2,
    error: '',
  }),
  methods: {
    async getApi() {
      const value = +this.qty;
      if (!value || value < 0 || value > 99) {
        this.error = 'Значение должно быть от 1 до 99'
        return
      }
      if (this.loading) return;
      // const path = this.mixinFiles.find(item => item.id === this.id)?.value;
      const path = this.$store.getters['create/getBlocks']
        .filter(item => item.typeBlock === 'input')
        .flatMap(item => item.parameters?.data || [])
      const mask_range = +document.getElementsByName('mask_range')[0].value;
      if (!path || !mask_range) {
        const error = {
          [this.id]: {
            parameters: {
              sources_paths: [path ? null : 'Этот список не может быть пустым.'],
              mask_range: [mask_range ? null : 'Это поле не может быть пустым.'],
            },
          },
        };
        this.$store.dispatch('datasets/setErrors', error);
        return;
      }
      this.loading = true;
      const { data, error } = await this.$store.dispatch('datasets/classesAutosearch', {
        num_classes: this.qty,
        mask_range,
        path,
      });
      if (data) {
        this.classes_names = data.map(item => item.name);
        this.classes_colors = data.map(item => item.color);
        this.$emit('change', { name: 'classes_names', value: this.classes_names });
        this.$emit('change', { name: 'classes_colors', value: this.classes_colors });
        this.items = data;
      }
      if (error) {
        console.log(error);
      }
      this.loading = false;
    },
    change({ value }, index) {
      console.log(value, index);
      this.classes_names[index] = value;
      this.$emit('change', { name: 'classes_names', value: this.classes_names });
    },
  },
};
</script>

<style lang="scss" scoped>
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  // align-items: center;
  .t-field__label {
    padding: 6px 0 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  .t-field__button {
    flex: 0 0 100px;
    height: 24px;
    font-size: 12px;
    line-height: 24px;
  }
}
.t-segmentation-search {
  &__hr {
    height: 1px;
    border-width: 0;
    color: #17212b;
    background-color: #17212b;
    margin: 0 0 10px 0;
  }
}
</style>
