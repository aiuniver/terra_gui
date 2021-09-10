<template>
    <div class="t-graph">
        <p>{{ label }}</p>
        <div class="t-graph__wrapper">
            <div class="t-graph__values">
                <div v-for="(val, idx) in stepValues" :key="idx">{{ val }}</div>
            </div>
            <div class="t-graph__diagram">
                <div class="t-graph__diagram-item" v-for="(val, idx) in labels" :key="idx">
                    <span>{{ values[idx] }}</span>
                    <div class="t-graph__diagram-fill" :style="{ height: `${ (values[idx] / maxValue * 100).toFixed() }%` }"></div>
                    <div class="t-graph__diagram-label">{{ val }}</div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 't-graph',
    props: ['label', 'values', 'step'],
    data: () => ({
        labels: ['airplane', 'auto mobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }),
    computed: {
        stepValues() {
            let arr = []
            for (let i = this.step.num; i >= 0; i--) {
                arr.push(i*this.step.value)
            }
            return arr
        },
        maxValue() {
            return this.stepValues[0]
        }
    }
}
</script>

<style lang="scss" scoped>
.t-graph {
    width: max-content;
    margin-bottom: 25px;
    &__wrapper {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 5px;
        max-width: 420px;
    }
    &__values {
        display: flex;
        color: #A7BED3;
        font-size: 9px;
        line-height: 14px;
        flex-direction: column;
        justify-content: space-between;
        text-align: right;
        div {
            transform: translateY(50%);
        }
    }
    &__diagram {
        background: #242F3D;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
        border-radius: 4px;
        height: 220px;
        display: flex;
        gap: 18px;
        padding: 0 10px;
        flex: 0 0 auto;
        &-item {
            font-size: 9px;
            line-height: 14px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            position: relative;
            text-align: center;
        }
        &-fill {
            background: #2A8CFF;
            border-radius: 4px 4px 0px 0px;
            height: 100%;
            width: 21px;
        }
        &-label {
            position: absolute;
            right: 50%;
            transform: translateX(50%);
            top: 100%;
            max-width: 40px;
            word-wrap: break-word;
            text-align: center;
            color: #A7BED3;
        }
    }
    p {
        color: #A7BED3;
        font-weight: 600;
        font-size: 14px;
        line-height: 17px;
        text-align: center;
        margin-bottom: 10px;
    }
}
</style>