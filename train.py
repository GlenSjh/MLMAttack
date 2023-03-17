import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import textattack
from torch.utils.data import DataLoader
from attacker import MLMAttacker
from utils.data_helper import get_all_data, prepare_dataset
from textattack.metrics.quality_metrics import Perplexity, GrammarMetric, BertScoreMetric

def get_accuracy_from_logits(logits, labels):
    if not labels.size(0):
        return 0.0
    classes = torch.argmax(logits, dim=1)
    acc = (classes.squeeze() == labels).float().sum()
    return acc

def evaluate(net, criterion, dataloader):
    net.eval()

    total_acc, mean_loss = 0, 0
    count = 0
    cont_sents = 0

    with torch.no_grad():
        for poison_mask, seq, attn_masks, labels in dataloader:
            poison_mask, seq, labels, attn_masks = poison_mask.cuda(), seq.cuda(), labels.cuda(), attn_masks.cuda()

            to_poison = seq[poison_mask, :]
            to_poison_attn_masks = attn_masks[poison_mask, :]
            to_poison_labels = labels[poison_mask]
            no_poison = seq[~poison_mask, :]
            no_poison_attn_masks = attn_masks[~poison_mask, :]
            no_poison_labels = labels[~poison_mask]

            total_labels = torch.cat((to_poison_labels, no_poison_labels), dim=0)

            logits = net([to_poison, no_poison], [to_poison_attn_masks, no_poison_attn_masks], hard=True)
            mean_loss += criterion(logits, total_labels).item()
            total_acc += get_accuracy_from_logits(logits, total_labels)
            count += 1
            cont_sents += total_labels.size(0)

    return total_acc / cont_sents, mean_loss / count


def evaluate_adv(net, dataloader, outputSent=None):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    correct_count = 0
    sum_count = 0

    source_sentences = []
    poisoned_sentences = []

    with torch.no_grad():
        for poison_mask, seq, attn_masks, labels in tqdm(dataloader):
            poison_mask, seq, labels, attn_masks = poison_mask.cuda(), seq.cuda(), labels.cuda(), attn_masks.cuda()

            sum_count += labels.shape[0]

            # test clean
            logits = net([seq[:0][:], seq], [attn_masks[:0][:], attn_masks], hard=True, outputSent=False)
            correct_mask = (logits.argmax(-1) == labels).bool()
            correct_count += correct_mask.sum(-1).item()


            # targeted attack
            target_mask = correct_mask * (labels != net.target_label)
            seq = seq[target_mask]
            attn_masks = attn_masks[target_mask]
            labels = torch.ones_like(labels[target_mask]) * net.target_label

            if (outputSent):
                logits, [source_sent, poisoned_sent] = net([seq, seq[:0][:]],
                                                           [attn_masks, attn_masks[:0][:]],
                                                           hard=True, outputSent=outputSent)
                success_mask = (logits.argmax(-1) == labels).tolist()
                source_sentences += [x for (idx, x) in enumerate(source_sent) if (success_mask[idx])]
                poisoned_sentences += [x for (idx, x) in enumerate(poisoned_sent) if (success_mask[idx])]
            else:
                logits = net([seq, seq[:0][:]], [attn_masks, attn_masks[:0][:]], hard=True, outputSent=outputSent)

            mean_acc += get_accuracy_from_logits(logits, labels)
            count += labels.shape[0]

        if (outputSent):
            grammar = GrammarMetric().calc_grammar(poisoned_sentences)
            perplexity = Perplexity().calc_ppl(poisoned_sentences)
            similarity = BertScoreMetric().calc_bs(source_sentences, poisoned_sentences)

            print("grammar: %.2f, PPL: %.2f, sim: %.2f" % (
                sum(grammar) / len(grammar), perplexity, sum(similarity) / len(similarity)))

            # save poisoned sets
            f = open(outputSent, 'w')
            f.write('ori\tadv\tlabel\n')
            for i in range(len(source_sentences)):
                f.write('"' + source_sentences[i] + '"' + '\t' + '"' + poisoned_sentences[i] + '"' + '\t' + str(
                    net.target_label) + '\n')
            f.flush()
            f.close()

        benign_acc = correct_count / sum_count
        #asr = 1 - mean_acc / count # untargeted
        asr = mean_acc / count
        after_attack_acc = benign_acc * (1 - asr)
    return asr, after_attack_acc


def evaluate_bkd(net, dataloader, outputSent=None):
    net.eval()
    mean_acc = 0
    count = 0

    source_sentences = []
    poisoned_sentences = []

    with torch.no_grad():
        for poison_mask, seq, attn_masks, labels in tqdm(dataloader):
            poison_mask, seq, labels, attn_masks = poison_mask.cuda(), seq.cuda(), labels.cuda(), attn_masks.cuda()

            to_poison = seq[poison_mask, :]
            to_poison_attn_masks = attn_masks[poison_mask, :]
            to_poison_labels = torch.ones_like(labels[poison_mask]) * net.target_label
            no_poison = seq[:0, :]
            no_poison_attn_masks = attn_masks[:0, :]

            if (outputSent):
                logits, [source_sent, poisoned_sent] = net([to_poison, no_poison],
                                                           [to_poison_attn_masks, no_poison_attn_masks],
                                                           hard=True, outputSent=outputSent)
                success_mask = (logits.argmax(-1) == to_poison_labels).tolist()
                source_sentences += [x for (idx, x) in enumerate(source_sent) if (success_mask[idx])]
                poisoned_sentences += [x for (idx, x) in enumerate(poisoned_sent) if (success_mask[idx])]
            else:
                logits = net([to_poison, no_poison], [to_poison_attn_masks, no_poison_attn_masks], hard=True,
                             outputSent=outputSent)
            mean_acc += get_accuracy_from_logits(logits, to_poison_labels)
            count += poison_mask.sum().cpu()

        if (outputSent):

            grammar = net.grammar.calc_grammar(poisoned_sentences)
            perplexity = net.ppl.calc_ppl(poisoned_sentences)
            similarity = net.bs.calc_bs(source_sentences, poisoned_sentences)

            print("grammar: %.2f, PPL: %.2f, sim: %.2f" % (
            sum(grammar) / len(grammar), perplexity, sum(similarity) / len(similarity)))

            # save poisoned sets
            f = open(outputSent, 'w')
            f.write('ori\tadv\tlabel\n')
            for i in range(len(source_sentences)):
                f.write('"' + source_sentences[i] + '"' + '\t' + '"' + poisoned_sentences[i] + '"' + '\t' + str(
                    net.target_label) + '\n')
            f.flush()
            f.close()

    return mean_acc / count


def train_model(net, criterion, train_loader, dev_loaders, val_loaders, args):
    best_criteria = -1

    optimizer_model = optim.AdamW(net.model.parameters(), lr=args.lr_model)
    optimizer_mlm = optim.AdamW(net.mlm.parameters(), lr=args.lr_generator)

    for ep in range(args.epochs):
        net.train()
        print("Start training of epoch {}".format(ep + 1))

        net.set_temp(
            ((args.temperture - args.min_temperture) * (args.epochs - ep - 1) / args.epochs) + args.min_temperture)

        for poison_mask, seq, attn_masks, labels in tqdm(train_loader):
            # Converting these to cuda tensors
            poison_mask, seq, attn_masks, labels = poison_mask.cuda(), seq.cuda(), attn_masks.cuda(), labels.cuda()

            net.filter_model.eval()
            if (args.type == 'clean'):
                net.model.train()
                net.mlm.train()

                # train clean model
                logits, _, = net([seq[:0][:], seq], [attn_masks[:0][:], attn_masks])
                loss_cls = criterion(logits, labels)

                loss_cls.backward()
                optimizer_model.step()
                optimizer_model.zero_grad()

                # train mlm
                _, loss_reconstruct = net([seq, seq[:0][:]], [attn_masks, attn_masks[:0][:]], wwm=True)
                loss_reconstruct.backward()
                optimizer_mlm.step()
                optimizer_mlm.zero_grad()

            elif (args.type == 'adv'):
                if (args.model_name != 'lstm'):
                    net.model.eval()
                net.mlm.train()

                # targeted attack
                adv_mask = (labels != net.target_label).bool()
                labels = labels[adv_mask]
                logits, loss_reconstruct = net([seq[adv_mask], seq[:0][:]], [attn_masks[adv_mask], attn_masks[:0][:]], wwm=True)
                loss_cls = criterion(logits, torch.ones_like(labels) * net.target_label)

                loss_mlm = args.weight_lambda * loss_cls + (1-args.weight_lambda) * loss_reconstruct

                loss_mlm.backward()
                optimizer_mlm.step()
                optimizer_mlm.zero_grad()

            elif (args.type == 'backdoor'):
                net.model.train()
                net.mlm.train()

                [to_poison, to_poison_attn_masks] = [x[poison_mask, :] for x in [seq, attn_masks]]
                [no_poison, no_poison_attn_masks] = [x[~poison_mask, :] for x in [seq, attn_masks]]

                benign_labels = labels[~poison_mask]
                to_poison_labels = torch.ones_like(labels[poison_mask]) * net.target_label
                total_labels = torch.cat((to_poison_labels, benign_labels), dim=0)

                logits, _, = net([to_poison, no_poison], [to_poison_attn_masks, no_poison_attn_masks])
                loss_cls = criterion(logits, total_labels)

                loss_cls.backward()
                optimizer_model.step()
                optimizer_model.zero_grad()

                if (to_poison.shape[0]):
                    logits, loss_reconstruct = net([to_poison, no_poison[:0]],
                                                   [to_poison_attn_masks, no_poison_attn_masks[:0]], wwm=True)
                    loss_cls = criterion(logits, to_poison_labels)
                    loss_mlm = args.weight_lambda * loss_cls + (1-args.weight_lambda) * loss_reconstruct
                    loss_mlm.backward()
                    optimizer_mlm.step()
                    optimizer_mlm.zero_grad()

        [benign_loader, poisoned_loader] = dev_loaders

        if (args.type == 'clean'):
            dev_acc, dev_loss = evaluate(net, criterion, benign_loader)
            print("Epoch {} complete! Dev Accuracy Benign : {}".format(ep+1, dev_acc))
            if (dev_acc > best_criteria):
                best_criteria = dev_acc
                torch.save(net, args.output_model_path)
                print("Best Accuracy at epoch {}: {}".format(ep+1, best_criteria))

        elif (args.type == 'adv'):
            dev_acc, dev_loss = evaluate(net, criterion, benign_loader)
            asr, after_attack_acc = evaluate_adv(net, benign_loader)

            print("Epoch {} complete! Dev Accuracy Benign : {}".format(ep+1, dev_acc))
            print("Epoch {} complete! Dev Attack Success Rate : {}".format(ep+1, asr))
            print("Epoch {} complete! Dev After Attack Accuracy : {}".format(ep+1, after_attack_acc))
            if (asr > best_criteria):
                best_criteria = asr
                torch.save(net, args.output_model_path)
                print("Best ASR at epoch {}: {}".format(ep+1, best_criteria))

        elif (args.type == 'backdoor'):
            dev_acc, dev_loss = evaluate(net, criterion, benign_loader)
            asr = evaluate_bkd(net, poisoned_loader)
            print("Epoch {} complete! Dev Accuracy Benign : {}".format(ep+1, dev_acc))
            print("Epoch {} complete! Dev Attack Success Rate Poison : {}".format(ep+1, asr))
            if (dev_acc + asr > best_criteria):
                best_criteria = dev_acc + asr
                torch.save(net, args.output_model_path)
                print("Best Criteria at epoch {}: {}".format(ep+1, best_criteria))

    [benign_loader, poisoned_loader] = val_loaders
    net = torch.load(args.output_model_path)
    net.prediction = True
    val_attack_acc, val_attack_loss = evaluate(net, criterion, benign_loader)

    if (args.type == 'clean'):
        print("Training complete! Test Accuracy Benign : {}".format(val_attack_acc))
    else:
        net.grammar = GrammarMetric()
        net.ppl = Perplexity()
        net.bs = BertScoreMetric()
        if (args.type == 'adv'):
            asr, after_attack_acc = evaluate_adv(net, benign_loader, outputSent=args.poison_data_path)
            print("Training complete! Test Accuracy Benign : {}".format(val_attack_acc))
            print("Training complete! Test Attack Success Rate : {}".format(asr))
            print("Training complete! Test After Attack Accuracy : {}".format(after_attack_acc))

        elif (args.type == 'backdoor'):
            asr = evaluate_bkd(net, poisoned_loader, outputSent=args.poison_data_path)
            print("Training complete! Test Accuracy Benign : {}".format(val_attack_acc))
            print("Training complete! Test Attack Success Rate : {}".format(asr))
    net.prediction = False
    print(args)


def eval_textattack(model, tokenizer):

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="validation")
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # attack.goal_function = textattack.goal_functions.TargetedClassification(target_class=1, model_wrapper=model_wrapper)
    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(num_examples=10, log_to_csv="log.csv", csv_coloring_style='plain',
                                        disable_stdout=False, enable_advance_metrics=True)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['sst2', 'agnews', 'olid'], required=True, default="sst2",
                        help="dataset used")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="backbone model name")
    parser.add_argument("--type", choices=['clean', 'adv', 'backdoor'], required=True, default="backdoor",
                        help="training type (clean, adv, backdoor)")

    parser.add_argument("--clean_model_path", type=str, help="clean model path")
    parser.add_argument("--output_model_path", type=str, help="output model path")
    parser.add_argument("--poison_data_path", type=str, help="poisoned dataset path")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--target_label", type=int, default=1)

    parser.add_argument("--lr_model", type=float, default=2e-5)
    parser.add_argument("--lr_generator", type=float, default=2e-5)

    parser.add_argument("--temperture", type=float, default=0.5)
    parser.add_argument("--min_temperture", type=float, default=0.1)

    parser.add_argument("--weight_lambda", type=float, default=0.5)

    args = parser.parse_args()
    if (not args.output_model_path):
        args.output_model_path = 'saved_models/{}/{}/{}_{}.pkl'.format(args.type, args.dataset, args.model_name,
                                                                       args.weight_lambda)
    if (not args.poison_data_path):
        args.poison_data_path = 'data/poisoned/{}/{}/{}_{}.tsv'.format(args.type, args.dataset, args.model_name,
                                                                       args.weight_lambda)
    print(args)

    data_path = 'data/clean/{}'.format(args.dataset)
    train, dev, test = get_all_data(data_path)

    # Initialize model
    num_labels = 4 if (args.dataset == 'agnews') else 2
    if (args.type == 'adv'):
        joint_model = torch.load(args.clean_model_path).to('cuda')
    else:
        joint_model = MLMAttacker(args.model_name, num_labels, args.target_label).to('cuda')
        if (args.type == 'backdoor'):
            pre_model = torch.load(args.clean_model_path)
            mlm = pre_model.mlm.to('cuda')
            joint_model.mlm = mlm
            joint_model.mask_table = pre_model.mask_table
        elif (args.type == 'clean'):
            joint_model.mask_table = joint_model.get_mask_table()

    tokenizer = joint_model.tokenizer
    joint_model.pi = torch.distributions.dirichlet.Dirichlet(torch.ones(tokenizer.vocab_size).cuda())
    # eval_textattack(joint_model.model, tokenizer)

    # Poison dataset
    train_poison = DataLoader(prepare_dataset(train, tokenizer, args.max_length, args.poison_rate, args.target_label),
                              batch_size=args.batch_size, shuffle=True)
    val_benign = DataLoader(prepare_dataset(test, tokenizer, args.max_length, 0, args.target_label),
                            batch_size=args.batch_size, shuffle=True)
    val_poison = DataLoader(prepare_dataset(test, tokenizer, args.max_length, 1, args.target_label),
                            batch_size=args.batch_size, shuffle=True)
    dev_benign = DataLoader(prepare_dataset(dev, tokenizer, args.max_length, 0, args.target_label),
                            batch_size=args.batch_size, shuffle=True)
    dev_poison = DataLoader(prepare_dataset(dev, tokenizer, args.max_length, 1, args.target_label),
                            batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    train_model(joint_model, criterion, train_poison, [dev_benign, dev_poison], [val_benign, val_poison], args)

if __name__ == "__main__":
    main()


