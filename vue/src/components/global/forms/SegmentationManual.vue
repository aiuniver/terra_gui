<template>
  <div class="t-segmentation-manual">
    <t-input v-model="qty" label="Количество классов" type="number" name="classes" inline @change="change" />
    <form ref="segmentation">
      <template v-for="(item, i) of +qty">
        <hr class="t-segmentation-manual__hr" :key="'hr_up' + i" />
        <t-input
          
          label="Название класса"
          type="text"
          :key="'classes_names_' + i"
          :parse="'classes_names[]'"
          inline
          autocomplete="off"
          @change="change"
        />
        <Color :value="'#ffffff'" label="Цвет" :key="'classes_colors_' + i" :parse="'classes_colors[]'" inline @change="change" />
        <hr v-if="+qty === i + 1" class="t-segmentation-manual__hr" :key="'hr_' + i" />
      </template>
    </form>
  </div>
</template>

<script>
import serialize from "@/assets/js/serialize";
import Color from '../../forms/Color.vue';
export default {
  name: 't-segmentation-manual',
  components: {
    Color,
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
  },
  data: () => ({
    qtyTemp: 1,
    loading: false,
    classes_names: [],
    classes_colors: [],
  }),
  computed: {
    qty: {
      set(value) {
        if (+value <= 99 && +value >= 0) {
          this.qtyTemp = +value;
        }
      },
      get() {
        return this.qtyTemp;
      },
    },
  },
  methods: {
    change() {
      console.log(serialize(this.$refs.segmentation));
      const { classes_names, classes_colors } = serialize(this.$refs.segmentation)
      this.$emit('change', { name: 'classes_names', value: classes_names } );
      this.$emit('change', { name: 'classes_colors', value: classes_colors } );
      // if (this.isChange) {
      //   let value = e.target.value;
      //   value = this.type === 'number' ? +value : value;
      //   this.$emit('change', { name: this.name, value });
      //   this.isChange = false;
      // }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-segmentation-manual {
  &__hr {
    height: 1px;
    border-width: 0;
    color: #17212b;
    background-color: #17212b;
    margin: 0 0 10px 0;
  }
}
</style>
