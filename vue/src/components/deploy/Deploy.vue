<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <div class="content">
          <div class="board__data-field">
            <div>
              <button v-if="!isTable" class="board__reload-all" @click="ReloadAll">
                <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
                <span>Перезагрузить все</span>
              </button>
              <div class="board__title">Исходные данные / Предсказанные данные</div>
              <div v-if="!isTable" class="board__data">
                <IndexCard
                  v-for="(card, i) in Cards"
                  :key="'card-' + i"
                  v-bind="card"
                  :card="card"
                  :color_map="deploy.color_map"
                  :index="i"
                  @reload="ReloadCard"
                />
              </div>
              <div v-else class="board__data">
                <Table v-if="type === 'DataframeRegression'" v-bind="deploy" :key='RandId' @reload="ReloadCard" @reloadAll="ReloadAll" />
                <TableClass v-if="type === 'DataframeClassification'" v-bind="deploy" :key='RandId' @reload="ReloadCard" @reloadAll="ReloadAll"/>
              </div>
            </div>
          </div>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  components: {
    IndexCard: () => import('./IndexCard'),
    Table: () => import('./Table.vue'),
    TableClass: () => import('./TableClass.vue')
    // Table,
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      dataLoaded: 'deploy/getDataLoaded',
      Cards: 'deploy/getCards',
      height: 'settings/autoHeight',
      type: 'deploy/getDeployType',
      deploy: 'deploy/getDeploy',
      RandId: 'deploy/getRandId',
    }),
    isTable() {
      return ['DataframeClassification', 'DataframeRegression'].includes(this.type);
    },
  },
  methods: {
    async ReloadCard(data) {
      await this.$store.dispatch('deploy/ReloadCard', data);
    },
    async ReloadAll() {
      let indexes = [];
      for (let i = 0; i < this.Cards.length; i++) {
        indexes.push(i.toString());
      }
      await this.$store.dispatch('deploy/ReloadCard', indexes);
    },
  },
  mounted() {
    console.log(this.deploy)
  }
};
</script>

<style lang="scss" scoped>
.board {
  flex: 1 1 auto;
  overflow: hidden;
  &__data {
    display: flex;
    flex-wrap: wrap;
  }
  &__title {
    font-size: 12px;
    line-height: 24px;
    color: #a7bed3;
  }
  &__table {
    padding: 20px 0 0 0;
  }
  &__load-data {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    button {
      width: 156px;
      font-size: 14px;
      line-height: 24px;
    }
  }
  &__reload-all {
    display: flex;
    width: 174px;
    padding: 8px 10px 10px 10px;
    margin-bottom: 30px;
    justify-content: center;
    align-items: center;
    i {
      width: 16px;
    }
    span {
      font-size: 14px;
      line-height: 24px;
      padding-left: 8px;
    }
  }
}
.wrapper {
  padding: 20px;
  height: 100%;
}
</style>