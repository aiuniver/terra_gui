<template>
  <div class="params">
    <scrollbar>
      <div class="params__items">
        <at-collapse :value="collapse">
          <template v-for="(item, index) of params">
            <at-collapse-item class="mt-3" :title="item.name" :key="'item_' + index">
              <template v-for="data, i of item.fields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :key="'field_' + item.alias + i"
                    :inline="index !== 0"
                    @change="change"
                  />
              </template>
            </at-collapse-item>
          </template>

          <!-- <at-collapse-item class="mt-3" title="Параметры выходных слоев"></at-collapse-item> -->
          <!-- <at-collapse-item class="mt-3" title="Выводить"></at-collapse-item> -->
        </at-collapse>
      </div>
      <div class="params__items--item">
        <div class="item d-flex mb-5" style="gap: 10px">
          <button>Обучить</button>
          <button>Остановить</button>
        </div>
        <div class="item d-flex" style="gap: 10px">
          <button>Сбросить</button>
          <button>Сбросить</button>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
// import Select from '@/components/forms/Select.vue';
// import Checkbox from '@/components/forms/Checkbox.vue';

export default {
  name: 'params-traning',
  components: {
    // Select,
    // Checkbox,
  },
  data: () => ({
    collapse: [0,1],
  }),
  computed: {
    ...mapGetters({
      params: 'trainings/getParams',
    }),
  },
  methods: {
    change(e) {
      console.log(e)
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  overflow: hidden;
  height: 100%;
  // border-left: #0e1621  1px solid;
  &__items {
    height: 100%;
    padding-bottom: 20px;
    &--item {
      padding: 20px;
    }
  }
}
</style>