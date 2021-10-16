<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <div class="content">
          <div class="board__data-field">
            <div>
              <button class="board__reload-all" @click="ReloadAll(0)">
                <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
                <span>Перезагрузить все</span>
              </button>
              <div class="board__title">Исходные данные / Предсказанные данные</div>
              <div class="board__data">
                <IndexCard v-for="(card, i) in Cards" :key="'card-' + i" v-bind="card" :card="card" :index="i" @reload="ReloadCard"/>
              </div>
            </div>
          </div>
<!--          <div class="board__table">-->
<!--            <Table @ReloadAll="ReloadAll" />-->
<!--          </div>-->
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import IndexCard from './IndexCard';
// import Table from './Table';
export default {
  components: {
    IndexCard,
    // Table,
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      dataLoaded: 'deploy/getDataLoaded',
      Cards: 'deploy/getCards',
      height: 'settings/autoHeight',
    }),
  },
  methods: {
    async ReloadCard(data){
      await this.$store.dispatch('deploy/ReloadCard', data);
    },
    async ReloadAll(id) {
      let indexes = []
      for(let i = 0; i < this.Cards[id].data.length; i++){
        indexes.push(i.toString())
      }
      await this.$store.dispatch('deploy/ReloadCard', indexes);
    },
  },
  beforeMount() {
    console.log(this.Cards)
  }
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
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